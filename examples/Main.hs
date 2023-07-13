
module Main where

import qualified LLaMACPP as L
import Foreign.C.String (withCString)
import Foreign.Ptr (castPtr)

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

  welp :: L.ModelQuantizeParamsPtr <- castPtr <$> L.modelQuantizeDefaultParams

  print "default context params"

  _ <- L.contextDefaultParams -- wrap and return newtype

  print "loading model"

  -- model <- withCString
  --   "/home/dd/ai/models/open-llama-7b-v2-open-instruct.ggmlv3.q5_K_M.bin"
  --   (flip L.loadModelFromFile contextParams)

  -- print "freeing model"

  -- L.freeModel model



-- initialize model data structure
-- initialize context data structure

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
