module Main where

import Control.Exception (bracket)
import Data.Foldable (for_)
import Data.Functor ((<&>), void)
import Data.Maybe (fromJust)
import qualified Data.Vector.Storable as V
import qualified LLaMACPP as L
import Foreign.C.String (peekCString, withCString)
import Foreign.C.Types (CFloat, CInt)
import Foreign.ForeignPtr (newForeignPtr_)
import Foreign.Marshal.Alloc (free, malloc)
import Foreign.Marshal.Array (allocaArray, copyArray, newArray, mallocArray, peekArray, pokeArray, withArray)
import Foreign.Marshal.Utils (fromBool)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable(peek, poke))
import GHC.Conc (TVar, atomically, newTVarIO, readTVarIO, writeTVar)


--
-- ## Notes on main.cpp
--
-- This is basically the main loop logic, mapping to a single
--    prompt -> tokenize -> eval -> (sample -> eval)
-- pass, which doesn't really map to a single loop, and I'm
-- disregarding stuff like sessions and guidance and a bunch of things
-- that can be adjusted based on parameters. There is a lot of
-- logic also to enable interactive prompts so that relevant data is
-- fed in and "rotated" intelligently to maintain the context during a
-- session, and that is also disregarded here (for now):
--
-- * collect prompt information/user-input and tokenize, storing in `embd_inp`.
--   * `embd_inp` never appears to be cleared, so it just accumulates values as the program proceeds.
--   * https://github.com/ggerganov/llama.cpp/blob/e782c9e735f93ab4767ffc37462c523b73a17ddc/examples/main/main.cpp#L649-L733, although this ignores some other points where prompt data is stuffed into `embd_inp` ahead of the main loop, based on input parameters and whatnot
--
-- * store tokenized input in `embd` along with `last_n_tokens`
--   * `last_n_tokens` is reset so its size is maintained. [The size is n_ctx](https://github.com/ggerganov/llama.cpp/blob/e782c9e735f93ab4767ffc37462c523b73a17ddc/examples/main/main.cpp#L344).
--
-- * tokenized data in embd is eval'd, then sampled
--   * https://github.com/ggerganov/llama.cpp/blob/e782c9e735f93ab4767ffc37462c523b73a17ddc/examples/main/main.cpp#L497-L507
--   * after every token generated, the token is fed back in and eval'ed so sampling takes that into account on the next pass
--

data Params = Params
  { _nCtx :: CInt
  , _nThreads :: CInt
  , _nPredict :: CInt
  , _nGpuLayers :: CInt
  , _enableNumaOpts :: Bool
  , _modelPath :: Maybe String
  , _alphaFrequency :: CFloat
  , _alphaPresence :: CFloat
  , _penalizeNl :: Bool
  , _repeatPenalty :: CFloat
  , _topK :: CInt    -- <= 0 to use vocab size
  , _topP :: CFloat -- 1.0 = disabled
  , _tfsZ :: CFloat -- 1.0 = disabled
  , _typicalP :: CFloat -- 1.0 = disabled
  , _temp :: CFloat -- 1.0 = disabled
  }
  deriving (Eq, Show)

-- mostly from examples/common.h
defaultParams :: Params
defaultParams = Params
  { _nCtx = 2048
  , _nThreads = 8
  , _nPredict = 50
  , _nGpuLayers = 32
  , _enableNumaOpts = False
  , _modelPath = Nothing
  , _alphaFrequency = 0.0
  , _alphaPresence = 0.0
  , _penalizeNl = False
  , _repeatPenalty = 1.1
  , _topK = 40    -- <= 0 to use vocab size
  , _topP = 0.95 -- 1.0 = disabled
  , _tfsZ = 1.00 -- 1.0 = disabled
  , _typicalP = 1.00 -- 1.0 = disabled
  , _temp = 0.80 -- 1.0 = disabled
  }

main :: IO ()
main = do

  -- load/initialize params

  -- from https://huggingface.co/TheBloke/open-llama-7B-v2-open-instruct-GGML
  -- ./main
  --   -t 10        -- threads to run on
  --   -ngl 32      -- number of layers to store in VRAM
  --   -m open-llama-7b-v2-open-instruct.ggmlv3.q4_0.bin
  --   --color      -- colorize output
  --   -c 2048      -- size of prompt context
  --   --temp 0.7   -- temperature -- sampling tuning
  --   --repeat_penalty 1.1 -- sampling tuning
  --   -n -1 -- number of tokens to predict, -1 is infinity, just used
  --            to loop around in interactive mode
  --   -p "### Instruction: Write a story about llamas\n### Response:"
  -- (or interactive, instead of -p: -i -ins)

  let
    params = defaultParams
      { _modelPath =
          Just "/home/dd/code/ai/models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin"
      , _temp = 0.7
      }

  -- TODO:
  --   dump out build info
  --   generate default seed?

  bracket
     (initialize params)

     cleanup

     (\(_cpp, ctx, _model) -> do
         putStrLn "\nSystem Info:"
         putStrLn =<< peekCString =<< L.printSystemInfo

         -- tokenizing & eval

         -- "do one empty run to warm up the model"
         allocaArray 1 $ \(tmp :: Ptr L.Token) -> do
           bos <- L.tokenBos
           pokeArray tmp [bos]
           _evalRes <- L.eval ctx tmp 1 0 (_nThreads params)
           L.resetTimings ctx

         let
           testString = "Instruction: What is a great way to scare a chicken? Response:\n"
           maxTokens = 1024
           tokenize s tks addBos = L.tokenize ctx s tks (fromIntegral maxTokens) (fromBool addBos)

         tokenized <- allocaArray maxTokens $ \tokensPtr -> do
           tokenizedCount <- withCString testString $ \ts -> tokenize ts tokensPtr True

           putStrLn "\nPrompt"
           putStrLn $ testString <> "\n\n"

           putStrLn $ "\nTokenized " <> show tokenizedCount

           putStrLn $ "\nRunning first eval of entire prompt"
           _evalRes <- L.eval ctx tokensPtr tokenizedCount 0 (_nThreads params)

           peekArray (fromIntegral tokenizedCount) tokensPtr


         -- sampling

         putStrLn $ "\nsampling"

         let
           remainderLength = 64 - length tokenized

         -- we want to feed everything into the repetition penalizing algo
         lastNTokens :: TVar [L.Token] <- newTVarIO $
           replicate remainderLength 0 <> tokenized

         sample params ctx lastNTokens (length tokenized)
     )

  where
    sample :: Params -> L.Context -> TVar [L.Token] -> Int -> IO ()
    sample params ctx lastNTokens nPast =
      for_ [0..(_nPredict params - 1)] $ \pn -> do
        nVocab <- fromIntegral <$> L.nVocab ctx

        candidatesPPtr <- malloc
        logitsCopyPtr <- mallocArray nVocab

        -- logits are a multidimensional array:
        -- logits[_vocab][n_tokens], as best as I can tell from how
        -- it's used ("rows/cols" as described in the docstring
        -- seems backwards)
        logitsPtr <- L.getLogits ctx
        copyArray logitsCopyPtr logitsPtr nVocab
        logitsCopyFPtr <- newForeignPtr_ logitsCopyPtr

        let
          logitsCopy = V.unsafeFromForeignPtr0 logitsCopyFPtr nVocab

          candidates = [0..(nVocab - 1)] <&> \n ->
            L.TokenData (fromIntegral n) (V.unsafeIndex logitsCopy n) 0.0

        candidatesPtr <- newArray candidates

        let
          candidatesP =
            L.TokenDataArray candidatesPtr (fromIntegral nVocab) False

        poke candidatesPPtr candidatesP

        lastNTokens' <- readTVarIO lastNTokens

        -- penalties

        withArray lastNTokens' $ \(lastNTokensPtr :: Ptr L.Token) -> do
          let
            lastNTokensLen = fromIntegral . length $ lastNTokens'

          -- putStrLn $ "\nlastNRepeat: " <> show lastNRepeat
          -- putStrLn $ "\nlastNTokens': " <> show lastNTokensLen
          -- putStrLn $ "\nlastNTokens': " <> show lastNTokens'

          -- float nl_logit = logits[llama_token_nl()];
          _nlLogit <- V.unsafeIndex logitsCopy . fromIntegral <$> L.tokenNl

          --
          -- lastNTokensPtr should be a pointer just to the last
          -- set of tokens at the end of lastNTokens matching the
          -- count of lastNRepeat, right now it's the entire thing
          --
          L.sampleRepetitionPenalty
            ctx candidatesPPtr lastNTokensPtr lastNTokensLen (_repeatPenalty params)

          L.sampleFrequencyAndPresencePenalties
            ctx
            candidatesPPtr
            lastNTokensPtr
            lastNTokensLen
            (_alphaFrequency params)
            (_alphaPresence params)

          -- todo
          -- if (!penalize_nl) {
          --     logits[llama_token_nl()] = nl_logit;
          -- }
          -- insert nlLogit into logits at index tokenNl

          -- todo: other sampling methods
          -- id' <- L.sampleTokenGreedy ctx candidatesPPtr

          L.sampleTopK ctx candidatesPPtr (_topK params) 1
          L.sampleTailFree ctx candidatesPPtr (_tfsZ params) 1
          L.sampleTypical ctx candidatesPPtr (_typicalP params) 1
          L.sampleTopP ctx candidatesPPtr (_topP params) 1
          L.sampleTemperature ctx candidatesPPtr (_temp params)
          id' <- L.sampleToken ctx candidatesPPtr

          this <- peekCString =<< L.tokenToStr ctx id'

          putStr this

          void $ withArray [id'] $ \newTokenArrPtr ->
            L.eval
              ctx newTokenArrPtr 1 (fromIntegral nPast + fromIntegral pn + 1) (_nThreads params)

          atomically . writeTVar lastNTokens $
            drop 1 lastNTokens' <> [id']

        free candidatesPPtr
        free logitsCopyPtr


    initialize :: Params -> IO (Ptr L.ContextParams, L.Context, L.Model)
    initialize params = do
      putStrLn "\ninitBackend"
      L.initBackend (_enableNumaOpts params)

      -- todo
      -- putStrLn "\ndefault model quantize params"

      cpp <- malloc
      putStrLn "\ninit context params"
      L.contextDefaultParams cpp
      ctxParams' <- peek cpp

      let
        ctxParams = ctxParams'
          { L._nCtx = (_nCtx params)
          , L._nGpuLayers = (_nGpuLayers params)
          }

      poke cpp ctxParams

      putStrLn "\nloading model"

      model <- withCString
        -- (-_-;)
        (fromJust $ _modelPath params) $ flip L.loadModelFromFile cpp

      putStrLn "\nloading context"

      ctx <- L.newContextWithModel model cpp
      pure (cpp, ctx, model)


    cleanup :: (Ptr L.ContextParams, L.Context, L.Model) -> IO ()
    cleanup (cpp, ctx, model) = do
      putStrLn "\n\nfreeing context, model, context params"

      L.printTimings ctx

      L.free ctx
      L.freeModel model
      free cpp

      L.freeBackend
