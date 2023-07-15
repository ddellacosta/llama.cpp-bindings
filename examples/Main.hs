
module Main where

import qualified LLaMACPP as L
import Foreign.C.String (withCString)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Array (allocaArray)
import Foreign.Ptr (Ptr, castPtr, nullFunPtr, nullPtr)
import Foreign.Storable (Storable(poke))

main :: IO ()
main = do

-- load/initialize params
-- * determine what we want to be able to initialize
--   * n_ctx < 2048
-- * provide defaultParams

  -- modelQuantizeParams <- L.modelQuantizeDefaultParams


-- dump out build info
-- generate default seed
-- init_backend

  let enableNumaOpts = False

  print "initBackend"

  L.initBackend enableNumaOpts

  print "default model quantize params"

  _modelQuantParams :: L.ModelQuantizeParamsPtr <- castPtr <$> L.modelQuantizeDefaultParams

  --
  -- this causes a segfault, I assume because memory is not getting
  -- allocated properly, but I'm not good enough at C/Haskell FFI to
  -- figure it out quickly so punting for now and just defining my own
  -- ContextParams below
  --
  -- contextParams :: IO L.ContextParams <- peek . castPtr <$> L.contextDefaultParams

  alloca $ \(cpp :: L.ContextParamsPtr) ->
    allocaArray 1 $ \ap -> do

      print "default context params"

      let
        contextParams = L.ContextParams
          (-1) -- seed
          512 -- _nCtx
          512 -- _nBatch
          0 -- _nGpuLayers
          0 -- _mainGpu
          ap -- _tensorSplit
          nullFunPtr -- _progressCallback
          nullPtr -- _progressCallbackUserData
          False --_lowVRAM
          True -- _f16KV
          False -- _logitsAll
          False -- _vocabOnly
          True -- _useMmap
          False -- _useMlock
          False -- _embedding

      poke cpp contextParams

-- initialize model data structure
-- initialize context data structure

      print "loading model"

      model <- withCString
        "/home/dd/code/ai/models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin"
        (flip L.loadModelFromFile $ castPtr cpp)


--
-- llama_init_from_gpt_params
-- * part of llama.cpp examples common namespace
-- * initializes params that were passed as arguments
-- * internally calls
--   context_default_params
--   load_model_from_file
--   new_context_with_model (if fails calls free_model)
--   model_apply_lora_from_file if lora_adapter is not empty (free/free_model if fails on cleanup)
-- * returns a tuple with the model and context initialized
--

-- print system info

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

      print "freeing model"

      L.freeModel model
