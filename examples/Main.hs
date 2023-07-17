{-# LANGUAGE TemplateHaskell #-}

module Main where

import Control.Lens
import Control.Lens.TH (makeFieldsNoPrefix)
import qualified LLaMACPP as L
import Foreign.C.String (peekCString, withCString)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (allocaArray, pokeArray)
import Foreign.Marshal.Utils (with)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (Storable(peek, poke, sizeOf))

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
  --   --temp 0.7   -- temperature -- seems to be about tuning?
  --   --repeat_penalty 1.1 -- seems to be about tuning?
  --   -n -1 -- number of tokens to predict, -1 is infinity, just used
  --            to loop around in interactive mode?
  --   -p "### Instruction: Write a story about llamas\n### Response:"
  -- (or interactive, instead of -p: -i -ins)

  let
    nThreads = 10
    nCtx = 2048 -- 2048
    nGpuLayers = 32
    enableNumaOpts = False
    _nPredict = -1 -- do I need this?

  -- TODO:
  --   dump out build info
  --   generate default seed

  putStrLn "\ninitBackend"

  L.initBackend enableNumaOpts

  putStrLn "\ndefault model quantize params"

  _modelQuantParams :: L.ModelQuantizeParamsPtr <- castPtr <$> L.modelQuantizeDefaultParams

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

  allocaArray 1 $ \ap -> do
    putStrLn "\ninit context params"

    let
      contextParams =
        (L.defaultContextParams ap)
        { L._nCtx = nCtx
        , L._nGpuLayers = nGpuLayers
        }

    with contextParams $ \cpp -> do

      -- just to prove that it's actually in the freaking struct at
      -- that memory address
      newContextParams <- peek cpp

      putStrLn ""
      putStrLn $ "Seed: " <> (show $ L._seed newContextParams)
      putStrLn $ "nCtx: " <> (show $ L._nCtx newContextParams)

      putStrLn "\nloading model"

      model <- withCString
        "/home/dd/code/ai/models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin"
        (flip L.loadModelFromFile cpp)

      putStrLn "\nloading context"

      ctx <- L.newContextWithModel model cpp

      -- skipping Lora stuff until I can get a bare minimum POC up and running

      putStrLn "\nSystem Info:"

      -- * end common's llama_init_from_gpt_params functionality *

      -- print system info
      putStrLn =<< peekCString =<< L.printSystemInfo

      -- if mem_test .... todo

      -- if export_cgraph .... todo

      -- path_session = path_prompt_cache
      -- session_tokens -- what are these? Record of all strings passed back and forth?

      -- if path_session !empty
      -- * calls
      --   open session file
      --   session_tokens resized ?
      --   load_session_file
      --   again resize session_tokens
      --   prints stuff

      -- embd_inp - default prompt?

      -- if we are just getting started add the initial prompt to embd_inp?
      -- calls tokenize for this
      -- otherwise we default to session_tokens. Kinda thinking more and
      -- more session_tokens are the actual session strings

      -- if we have session_tokens
      --  ... todo, this bit (line 203 - 222)

      -- ... a bunch of prompt twiddling and debugging info? until 270

      -- logic for how the console works in interactive mode until 300, but
      -- before it's really started

      -- dumping out info about params

      -- last_n_tokens ?

      -- more interactive logic

      -- embd defined - input string?

      -- "do one empty run to warm up the model"
      -- * uses
      --   token_bos
      --   eval
      --   reset_timings

      allocaArray 1 $ \(tmp :: Ptr L.Token) -> do
        bos <- L.tokenBos
        pokeArray tmp [bos]
        L.eval ctx tmp 1 0 nThreads
        L.resetTimings ctx

-- while loop, main interactive loop
-- * flow/functions used
--   checks input size
--   "infinite text generation via context swapping"
--   - this seems like it's resetting the vector, keeping on
--     the first value and otherwise removing items by
--     discarding them out of the front of queue as new ones
--     are fed into the end. I guess this is supply the context?
--   "try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)"
--   - not sure what this is...todo. More session_token resizing
--   "evaluate tokens in batches"
--   then there's a big section that looks like cleanup when the loop
--   ends, based on checks (e.g. !is_interacting)
--     - save session optionally
--     - does a bunch of stats calculations?
--     - console newline fiddling
--   reverse prompt check? no idea ...todo
--   output for next line of interaction
--   end of text token
--   return user control when max tokens reached in output, I assume

      putStrLn "\nfreeing model"

      L.freeModel model
