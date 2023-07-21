module Main where

import Control.Exception (bracket, finally)
import Data.Foldable (foldlM, for_)
import Data.Functor ((<&>), void)
import qualified Data.Vector.Storable as V
import qualified LLaMACPP as L
import Foreign.C.String (peekCString, withCString)
import Foreign.C.Types (CFloat, CInt)
import Foreign.ForeignPtr (newForeignPtr_)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (allocaArray, copyArray, newArray, peekArray, pokeArray, withArray)
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
    nPredict = 50
    nThreads = 8
    nCtx = 2048 -- 2048
    nCtxInt = fromIntegral nCtx
    nGpuLayers = 32
    enableNumaOpts = False

  -- TODO:
  --   dump out build info
  --   generate default seed

  putStrLn "\ninitBackend"

  L.initBackend enableNumaOpts

  putStrLn "\ndefault model quantize params"

  --
  -- * start llama_init_from_gpt_params *
  --
  -- * part of llama.cpp examples common namespace
  -- * initializes params that were passed as arguments
  -- * internally calls
  --   context_default_params
  --   load_model_from_file
  --   new_context_with_model (if fails calls free_model)
  --   model_apply_lora_from_file if lora_adapter is not empty (free/free_model if fails on cleanup)
  -- * returns a tuple with the model and context initialized

  -- this is what is set in common.cpp's
  --   llama_context_params_from_gpt_params:

  -- lparams.n_ctx        = params.n_ctx;
  -- lparams.n_gpu_layers = params.n_gpu_layers;
  -- -- (from here below is todo)
  -- lparams.main_gpu     = params.main_gpu;
  -- lparams.n_batch      = params.n_batch;
  -- memcpy(lparams.tensor_split, params.tensor_split, LLAMA_MAX_DEVICES*sizeof(float));
  -- lparams.low_vram     = params.low_vram;
  -- lparams.seed         = params.seed;
  -- lparams.f16_kv       = params.memory_f16;
  -- lparams.use_mmap     = params.use_mmap;
  -- lparams.use_mlock    = params.use_mlock;
  -- lparams.logits_all   = params.perplexity;
  -- lparams.embedding    = params.embedding;

  alloca $ \(cpp :: L.ContextParamsPtr) -> do
    putStrLn "\ninit context params"
    L.contextDefaultParams cpp
    ctxParams' <- peek cpp

    let
      ctxParams = ctxParams'
        { L._nCtx = nCtx
        , L._nGpuLayers = nGpuLayers
        }

    poke cpp ctxParams

    putStrLn "\nloading model"

    model <- withCString
      "/home/dd/code/ai/models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin"
      (flip L.loadModelFromFile cpp)

    putStrLn "\nloading context"

    ctx <- L.newContextWithModel model cpp

    -- skipping Lora stuff until I can get a bare minimum POC up and running

    putStrLn "\nSystem Info:"

    putStrLn =<< peekCString =<< L.printSystemInfo

    -- tokenizing & eval

    -- "do one empty run to warm up the model"
    allocaArray 1 $ \(tmp :: Ptr L.Token) -> do
      bos <- L.tokenBos
      pokeArray tmp [bos]
      _evalRes <- L.eval ctx tmp 1 0 nThreads
      L.resetTimings ctx

    let
      testString = "Instruction: I have five pears, what recipe can I make? Response:\n"
      maxTokens = 1024
      tokenize s tks addBos = L.tokenize ctx s tks (fromIntegral maxTokens) (fromBool addBos)

    tokenized <- allocaArray maxTokens $ \tokensPtr -> do
      tokenizedCount <- withCString testString $ \ts -> tokenize ts tokensPtr True

      putStrLn $ "\nTokenized " <> show tokenizedCount

      _evalRes <- L.eval ctx tokensPtr tokenizedCount 0 nThreads

      putStrLn "\ntokenized, eval'ed our string"

      peekArray (fromIntegral tokenizedCount) tokensPtr


    -- sampling

    let
      remainderLength = 64 - length tokenized

    -- we want to feed everything into the repetition penalizing algo
    lastNTokens :: TVar [L.Token] <- newTVarIO $
      replicate remainderLength 0 <> tokenized

    putStrLn "\n"
    putStrLn $ testString <> "\n\n"

    finally
      (sample ctx (fromIntegral nCtx) nPredict nThreads lastNTokens (length tokenized))
      (cleanup ctx model)

  where
    sample :: L.Context -> Int -> Int -> CInt -> TVar [L.Token] -> Int -> IO ()
    sample ctx nCtx nPredict nThreads lastNTokens nPast =
      for_ [0..(nPredict - 1)] $ \pn ->
        alloca $ \candidatesPPtr -> do
          nVocab <- L.nVocab ctx
          let nVocabInt = fromIntegral nVocab
          allocaArray nVocabInt $ \logitsCopyPtr -> do
            -- logits are a multidimensional array:
            -- logits[_vocab][n_tokens], as best as I can tell from how
            -- it's used ("rows/cols" as described in the docstring
            -- seems backwards)
            logitsPtr <- L.getLogits ctx
            copyArray logitsCopyPtr logitsPtr nVocabInt
            logitsCopyFPtr <- newForeignPtr_ logitsCopyPtr

            let
              logitsCopy = V.unsafeFromForeignPtr0 logitsCopyFPtr nVocabInt

              candidates = [0..(nVocabInt - 1)] <&> \n ->
                L.TokenData (fromIntegral n) (V.unsafeIndex logitsCopy n) 0.0

            candidatesPtr <- newArray candidates

            let
              candidatesP =
                L.TokenDataArray candidatesPtr (fromIntegral nVocab) False

            poke candidatesPPtr candidatesP

            lastNTokens' <- readTVarIO lastNTokens

            -- penalties
            let
              alphaFrequency = 0.0 :: CFloat
              alphaPresence = 0.0 :: CFloat
              penalizeNl = False
              repeatPenalty = 1.1

              -- sampling defaults from common.h
              topK = 40    -- <= 0 to use vocab size
              topP = 0.95 -- 1.0 = disabled
              tfsZ = 1.00 -- 1.0 = disabled
              typicalP = 1.00 -- 1.0 = disabled
              temp = 0.80 -- 1.0 = disabled

              lastNTokensLen = fromIntegral . length $ lastNTokens'

            withArray lastNTokens' $ \(lastNTokensPtr :: Ptr L.Token) -> do
              -- putStrLn $ "\nlastNRepeat: " <> show lastNRepeat
              -- putStrLn $ "\nlastNTokens': " <> show lastNTokensLen
              -- putStrLn $ "\nlastNTokens': " <> show lastNTokens'

              -- float nl_logit = logits[llama_token_nl()];
              _nlLogit <- V.unsafeIndex logitsCopy . fromIntegral <$> L.tokenNl

              -- llama_sample_repetition_penalty(ctx, &candidates_p,
              --     last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
              --     last_n_repeat, repeat_penalty);

              -- lastNTokensPtr should be a pointer just to the last
              -- set of tokens at the end of lastNTokens matching the
              -- count of lastNRepeat
              L.sampleRepetitionPenalty ctx candidatesPPtr lastNTokensPtr lastNTokensLen repeatPenalty

              --  llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
              --      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
              --      last_n_repeat, alpha_frequency, alpha_presence);

              L.sampleFrequencyAndPresencePenalties
                ctx candidatesPPtr lastNTokensPtr lastNTokensLen alphaFrequency alphaPresence

              -- if (!penalize_nl) {
              --     logits[llama_token_nl()] = nl_logit;
              -- }
              -- insert nlLogit into logits at index tokenNl

              -- todo: other sampling methods
              -- id' <- L.sampleTokenGreedy ctx candidatesPPtr

              L.sampleTopK ctx candidatesPPtr topK 1
              L.sampleTailFree ctx candidatesPPtr tfsZ 1
              L.sampleTypical ctx candidatesPPtr typicalP 1
              L.sampleTopP ctx candidatesPPtr topP 1
              L.sampleTemperature ctx candidatesPPtr temp
              id' <- L.sampleToken ctx candidatesPPtr

              this <- peekCString =<< L.tokenToStr ctx id'

              putStr $ this <> " "

              void $ withArray [id'] $ \newTokenArrPtr ->
                L.eval ctx newTokenArrPtr 1 (fromIntegral nPast + fromIntegral pn + 1) nThreads

              atomically . writeTVar lastNTokens $
                drop 1 lastNTokens' <> [id']

    cleanup :: L.Context -> L.Model -> IO ()
    cleanup ctx model = do
      putStrLn "\n\nfreeing model"

      L.printTimings ctx

      L.free ctx
      L.freeModel model

      L.freeBackend
