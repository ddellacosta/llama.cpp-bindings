module Main where

import Control.Applicative ((<**>))
import Control.Exception (bracket)
import Control.Monad (when)
import Data.Foldable (for_)
import Data.Functor ((<&>), void)
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
import Options.Applicative
  ( ParseError(..)
  , Parser
  , auto
  , execParser
  , fullDesc
  , help
  , helper
  , info
  , long
  , noArgError
  , option
  , progDesc
  , short
  , showDefault
  , strOption
  , switch
  , value
  )

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
  , _prompt :: String
  , _modelPath :: String
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

paramsParser :: Parser Params
paramsParser = Params
  <$> option auto ( long "n_ctx" <> value 512 <> showDefault <> help "context size" )
  <*> option auto ( long "n_threads" <> short 't' <> value 1 <> showDefault <> help "cpu physical cores" )
  -- the llama.cpp options set -1 to be infinite generation here,
  -- but I'd rather not use that convention...but haven't
  -- determined what I want to replace it with yet.
  <*> option auto ( long "n_predict" <> value 50 <> showDefault <> help "new tokens to predict" )
  <*> option auto (
        long "n_gpu_layers" <>
        value 0 <>
        showDefault <>
        help "number of layers to store in VRAM"
      )
  <*> switch (
        long "enable_numa_opts" <>
        help "attempt optimizations that help on some NUMA systems"
      )
  <*> strOption (
       long "prompt" <>
       short 'p' <>
       help "prompt" <>
       (noArgError . ExpectsArgError $ "you must supply a prompt")
     )
  <*> strOption (
       long "model_path" <>
       short 'm' <>
       help "path to model" <>
       (noArgError . ExpectsArgError $ "you must supply a model path")
     )
  <*> option auto ( long "alpha_freq" <> value 0.0 <> showDefault <> help "" )
  <*> option auto ( long "alpha_pres" <> value 0.0 <> showDefault <> help "" )
  <*> switch ( long "penalize_nl" <> showDefault <> help "" )
  <*> option auto ( long "repeat_penalty" <> value 1.1 <> showDefault <> help "" )
  <*> option auto ( long "top_k" <> value 40 <> showDefault <> help "" )
  <*> option auto ( long "top_p" <> value 0.95 <> showDefault <> help "" )
  <*> option auto ( long "tfs_z" <> value 1.0 <> showDefault <> help "" )
  <*> option auto ( long "typical_p" <> value 1.0 <> showDefault <> help "" )
  <*> option auto ( long "temp" <> value 0.8 <> showDefault <> help "" )

-- mimicking call here, somewhat: https://huggingface.co/TheBloke/open-llama-7B-v2-open-instruct-GGML#how-to-run-in-llamacpp 
-- result/bin/examples +RTS -xc -RTS -m ../models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin --n_ctx 2048 --temp 0.7 -t 8 --n_gpu_layers 32 -p "what is a good tomato recipe?"


main :: IO ()
main = do

  -- load/initialize params

  let
    opts = info (paramsParser <**> helper)
      ( fullDesc
     <> progDesc "run a single prompt (for now) against a model"
      )

  params <- execParser opts

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
           maxTokens = 1024
           tokenize s tks addBos =
             L.tokenize ctx s tks (fromIntegral maxTokens) (fromBool addBos)

         tokenized <- allocaArray maxTokens $ \tokensPtr -> do
           tokenizedCount <- withCString (_prompt params) $ \ts -> tokenize ts tokensPtr True

           putStrLn "\nPrompt"
           putStrLn $ (_prompt params) <> "\n\n"

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
    sample params ctx lastNTokens nPast = do
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

      -- sampling

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
          ctx newTokenArrPtr 1 (fromIntegral nPast) (_nThreads params)

      atomically . writeTVar lastNTokens $
        drop 1 lastNTokens' <> [id']

      free candidatesPPtr
      free logitsCopyPtr

      eos <- L.tokenEos
      when (eos /= id') $
        sample params ctx lastNTokens (nPast + 1)


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

      model <- withCString (_modelPath params) $ flip L.loadModelFromFile cpp

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
