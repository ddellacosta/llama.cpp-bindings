
module LLaMACPP where

import Data.Either (fromRight)
import Control.Exception (assert)
import Data.Word (Word32)
import Foreign
import Foreign.C.String (peekCStringLen)
import Foreign.C.Types (CChar, CDouble, CFloat, CInt, CLong, CSize, CUChar, CULong)
import Foreign.Marshal.Utils (fromBool, toBool)
import Foreign.Ptr (FunPtr, Ptr, castPtr)
import Foreign.Storable (Storable, alignment, peek, poke, sizeOf)

#include "llama.h"
-- this is here so I can pass structs by reference to C API functions
-- that expect them by value
#include "wrapper.h"

{# context lib="llama" prefix="llama" #}


--
-- struct llama_model;
--
{# pointer *model as Model newtype #}

instance Storable Model where
  sizeOf (Model t) = sizeOf t
  alignment (Model t) = alignment t
  peek p = Model <$> (peek . castPtr $ p)
  poke p (Model t) = poke (castPtr p) t


--
-- struct llama_context;
--
-- `context` alone conflicts with the c2hs keyword
{# pointer *llama_context as Context newtype #}

instance Storable Context where
  sizeOf (Context t) = sizeOf t
  alignment (Context t) = alignment t
  peek p = Context <$> (peek . castPtr $ p)
  poke p (Context t) = poke (castPtr p) t


--
-- typedef int llama_token;
--
type Token = {# type token #}
  

--
-- typedef struct llama_token_data {
--     llama_token id; // token id
--     float logit;    // log-odds of the token
--     float p;        // probability of the token
-- } llama_token_data;
--
data TokenData = TokenData
  { _id :: Token
  , _logit :: CFloat
  , _p :: CFloat
  }
  deriving (Eq, Show)

{# pointer *token_data as TokenDataPtr -> TokenData #}

instance Storable TokenData where
  sizeOf _ = {# sizeof token_data #}
  alignment _ = {# alignof token_data #}
  peek p = TokenData
    <$> {# get token_data->id #} p
    <*> {# get token_data->logit #} p
    <*> {# get token_data->p #} p
  poke p (TokenData _id _logit _p) = do
    {# set token_data->id #} p _id
    {# set token_data->logit #} p $ _logit 
    {# set token_data->p #} p $ _p


--
-- typedef struct llama_token_data_array {
--     llama_token_data * data;
--     size_t size;
--     bool sorted;
-- } llama_token_data_array;
--
data TokenDataArray = TokenDataArray
  { _data :: Ptr TokenData
  , _size :: CSize
  , _sorted :: Bool
  }
  deriving (Eq, Show)

{# pointer *token_data_array as TokenDataArrayPtr -> TokenDataArray #}

instance Storable TokenDataArray where
  sizeOf _ = {# sizeof token_data_array #}
  alignment _ = {# alignof token_data_array #}
  peek p = TokenDataArray
    <$> {# get token_data_array->data #} p
    <*> (fromIntegral <$> {# get token_data_array->size #} p)
    <*> {# get token_data_array->sorted #} p
  poke p (TokenDataArray _data _size _sorted) = do
    {# set token_data_array->data #} p _data
    {# set token_data_array->size #} p $ fromIntegral _size
    {# set token_data_array->sorted #} p _sorted

--
-- typedef void (*llama_progress_callback)(float progress, void *ctx);
--
type ProgressCallback = CFloat -> Ptr () -> IO ()

{# pointer progress_callback as ProgressCallbackPtr -> ProgressCallback #}

--     struct llama_model_params {
--         int32_t n_gpu_layers; // number of layers to store in VRAM
--         int32_t main_gpu;     // the GPU that is used for scratch and small tensors
--         const float * tensor_split; // how to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
-- 
--         // called with a progress value between 0 and 1, pass NULL to disable
--         llama_progress_callback progress_callback;
-- 
--         // context pointer passed to the progress callback
--         void * progress_callback_user_data;
-- 
--         // override key-value pairs of the model meta data
--         const struct llama_model_kv_override * kv_overrides;
-- 
--         // Keep the booleans together to avoid misalignment during copy-by-value.
--         bool vocab_only; // only load the vocabulary, no weights
--         bool use_mmap;   // use mmap if possible
--         bool use_mlock;  // force system to keep model in RAM
--     };
data ModelParams = ModelParams
    { _nGpuLayers :: Word32
    , _mainGpu :: Word32
    , _tensorSplit :: Ptr CFloat
    , _progressCallback :: FunPtr ProgressCallback
    , _progressCallbackUserData :: Ptr ()
    , _kvOverrides :: Ptr ()
    , _vocabOnly :: Bool
    , _useMmap :: Bool
    , _useMlock :: Bool
    }
    deriving (Show, Eq)

{# pointer *model_params as ModelParamsPtr -> ModelParams #}

instance Storable ModelParams where
  sizeOf _ = {# sizeof model_params #}
  alignment _ = {# alignof model_params #}
  peek p = ModelParams
    <$> (fromIntegral <$> {# get model_params->n_gpu_layers #} p)
    <*> (fromIntegral <$> {# get model_params->main_gpu #} p)
    <*> ({# get model_params->tensor_split #} p)
    <*> ({# get model_params->progress_callback #} p)
    <*> ({# get model_params->progress_callback_user_data #} p)
    <*> ({# get model_params->kv_overrides #} p)
    <*> ({# get model_params->vocab_only #} p)
    <*> ({# get model_params->use_mmap #} p)
    <*> ({# get model_params->use_mlock #} p)
  poke p mps = do 
    {# set model_params->n_gpu_layers #} p $ fromIntegral $ _nGpuLayers mps
    {# set model_params->main_gpu #} p $ fromIntegral $ _mainGpu mps

--
--struct llama_context_params {
--     uint32_t seed;                         // RNG seed, -1 for random
--     int32_t  n_ctx;                        // text context
--     int32_t  n_batch;                      // prompt processing batch size
--     int32_t  n_gpu_layers;                 // number of layers to store in VRAM
--     int32_t  main_gpu;                     // the GPU that is used for scratch and small tensors
--     float tensor_split[LLAMA_MAX_DEVICES]; // how to split layers across multiple GPUs
--     // called with a progress value between 0 and 1, pass NULL to disable
--     llama_progress_callback progress_callback;
--     // context pointer passed to the progress callback
--     void * progress_callback_user_data;
--
--     // Keep the booleans together to avoid misalignment during copy-by-value.
--     bool low_vram;   // if true, reduce VRAM usage at the cost of performance
--     bool f16_kv;     // use fp16 for KV cache
--     bool logits_all; // the llama_eval() call computes all logits, not just the last one
--     bool vocab_only; // only load the vocabulary, no weights
--     bool use_mmap;   // use mmap if possible
--     bool use_mlock;  // force system to keep model in RAM
--     bool embedding;  // embedding mode only
-- };
--
data ContextParams = ContextParams
  { _seed :: Word32
  , _nCtx :: Int
  , _nBatch :: Int
  , _nThreads :: Word32
  , _nThreadsBatch :: Word32
  , _ropeScalingType :: Word8
  , _ropeFreqBase :: CFloat
  , _ropeFreqScale :: CFloat
  , _yarnExtFactor :: CFloat
  , _yarnAttnFactor :: CFloat
  , _yarnBetaFast :: CFloat
  , _yarnBetaSlow :: CFloat
  , _yarnOrigCtx :: Word32
  , _typeK :: CInt
  , _typeV :: CInt
  , _mulMatQ :: Bool
  , _logitsAll :: Bool
  , _embedding :: Bool
  , _offloadKqv :: Bool
  }
   deriving (Eq, Show)

{# pointer *context_params as ContextParamsPtr -> ContextParams #}

instance Storable ContextParams where
  sizeOf _ = {# sizeof context_params #}
  alignment _ = {# alignof context_params #}
  peek p = ContextParams
    <$> (fromIntegral <$> {# get context_params->seed #} p)
    <*> (fromIntegral <$> {# get context_params->n_ctx #} p)
    <*> (fromIntegral <$> {# get context_params->n_batch #} p)
    <*> (fromIntegral <$> {# get context_params->n_threads  #} p)
    <*> (fromIntegral <$> {# get context_params->n_threads_batch  #} p)
    <*> (fromIntegral <$> {# get context_params->rope_scaling_type  #} p)
    <*> {# get context_params->rope_freq_base  #} p
    <*> {# get context_params->rope_freq_scale  #} p
    <*> {# get context_params->yarn_ext_factor  #} p
    <*> {# get context_params->yarn_attn_factor  #} p
    <*> {# get context_params->yarn_beta_fast  #} p
    <*> {# get context_params->yarn_beta_slow  #} p
    <*> (fromIntegral <$> {# get context_params->yarn_orig_ctx  #} p)
    <*> {# get context_params->type_k  #} p
    <*> {# get context_params->type_k  #} p
    <*> {# get context_params->mul_mat_q  #} p
    <*> {# get context_params->logits_all  #} p
    <*> {# get context_params->embedding  #} p
    <*> {# get context_params->offload_kqv  #} p
    -- <*> (fromIntegral <$> {# get context_params->n_gpu_layers #} p)
    -- <*> (fromIntegral <$> {# get context_params->main_gpu #} p)
    -- <*> {# get context_params->tensor_split #} p
    -- <*> {# get context_params->progress_callback #} p
    -- <*> {# get context_params->progress_callback_user_data #} p
    -- <*> {# get context_params->low_vram #} p
    -- <*> {# get context_params->f16_kv #} p
    -- <*> {# get context_params->logits_all #} p
    -- <*> {# get context_params->vocab_only #} p
    -- <*> {# get context_params->use_mmap #} p
    -- <*> {# get context_params->use_mlock #} p
    -- <*> {# get context_params->embedding #} p
  poke p cps = do 
    {# set context_params->seed #} p $ fromIntegral $ _seed cps
    {# set context_params->n_ctx #} p $ fromIntegral $ _nCtx cps
    {# set context_params->n_batch #} p $ fromIntegral $ _nBatch cps
    {# set context_params-> n_threads  #} p $ fromIntegral $ _nThreads  cps
    {# set context_params-> n_threads_batch  #} p $ fromIntegral $ _nThreadsBatch  cps
    {# set context_params-> rope_scaling_type  #} p $ fromIntegral $ _ropeScalingType  cps
    {# set context_params-> rope_freq_base  #} p $ _ropeFreqBase  cps
    {# set context_params-> rope_freq_scale  #} p $ _ropeFreqScale  cps
    {# set context_params-> yarn_ext_factor  #} p $ _yarnExtFactor  cps
    {# set context_params-> yarn_attn_factor  #} p $ _yarnAttnFactor  cps
    {# set context_params-> yarn_beta_fast  #} p $ _yarnBetaFast  cps
    {# set context_params-> yarn_beta_slow  #} p $ _yarnBetaSlow  cps
    {# set context_params-> yarn_orig_ctx  #} p $ fromIntegral $ _yarnOrigCtx  cps
    {# set context_params-> type_k  #} p $ _typeK  cps
    {# set context_params-> type_v  #} p $ _typeV  cps
    {# set context_params-> mul_mat_q  #} p $ _mulMatQ  cps
    {# set context_params-> logits_all  #} p $ _logitsAll  cps
    {# set context_params-> embedding  #} p $ _embedding  cps
    {# set context_params-> offload_kqv  #} p $ _offloadKqv  cps
    -- {# set context_params->n_gpu_layers #} p $ fromIntegral $ _nGpuLayers cps
    -- {# set context_params->main_gpu #} p $ fromIntegral $ _mainGpu cps
    -- {# set context_params->tensor_split #} p $ _tensorSplit cps
    -- {# set context_params->progress_callback #} p $ _progressCallback cps
    -- {# set context_params->progress_callback_user_data #} p $ _progressCallbackUserData cps
    -- {# set context_params->low_vram #} p $ _lowVRAM cps
    -- {# set context_params->f16_kv #} p $ _f16KV cps
    -- {# set context_params->logits_all #} p $ _logitsAll cps
    -- {# set context_params->vocab_only #} p $ _vocabOnly cps
    -- {# set context_params->use_mmap #} p $ _useMmap cps
    -- {# set context_params->use_mlock #} p $ _useMlock cps
    -- {# set context_params->embedding #} p $ _embedding cps
 

--
-- // model file types
-- enum llama_ftype {
--     LLAMA_FTYPE_ALL_F32              = 0,
--     LLAMA_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
--     // LLAMA_FTYPE_MOSTLY_Q4_2       = 5, // support has been removed
--     // LLAMA_FTYPE_MOSTLY_Q4_3       = 6, // support has been removed
--     LLAMA_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q2_K          = 10,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17,// except 1d tensors
--     LLAMA_FTYPE_MOSTLY_Q6_K          = 18,// except 1d tensors
-- };
--
{#enum ftype as FType {underscoreToCase} deriving (Show, Eq)#}

--
-- // model quantization parameters
-- typedef struct llama_model_quantize_params {
--     int nthread;                 // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
--     enum llama_ftype   ftype;    // quantize to this llama_ftype
--     bool allow_requantize;       // allow quantizing non-f32/f16 tensors
--     bool quantize_output_tensor; // quantize output.weight
-- } llama_model_quantize_params;
--
data ModelQuantizeParams = ModelQuantizeParams
  { _nthread :: CInt
  , _ftype :: FType
  , _allowRequantize :: Bool
  , _quantizeOutputTensor :: Bool
  }
  deriving (Show, Eq)

{# pointer *model_quantize_params as ModelQuantizeParamsPtr -> ModelQuantizeParams #}

instance Storable ModelQuantizeParams where
  sizeOf _ = {# sizeof model_quantize_params #}
  alignment _ = {# alignof model_quantize_params #}
  peek p = ModelQuantizeParams
    <$> {# get model_quantize_params->nthread #} p
    <*> (toEnum . fromIntegral <$> {# get model_quantize_params->ftype #} p)
    <*> {# get model_quantize_params->allow_requantize #} p
    <*> {# get model_quantize_params->quantize_output_tensor #} p
  poke p (ModelQuantizeParams _nthread _ftype _ar _qot) = do
    {# set model_quantize_params->nthread #} p _nthread
    {# set model_quantize_params->ftype #} p (toEnum . fromEnum $ _ftype) -- lol
    {# set model_quantize_params->allow_requantize #} p _ar
    {# set model_quantize_params->quantize_output_tensor #} p _qot


--
-- LLAMA_API int llama_max_devices();
--
maxDevices :: IO CInt
maxDevices = {# call max_devices #}


--
-- LLAMA_API struct llama_context_params llama_context_default_params();
--
--
contextDefaultParams :: ContextParamsPtr -> IO ()
contextDefaultParams = {# call wrapper_context_default_params #}

modelDefaultParams :: ModelParamsPtr -> IO ()
modelDefaultParams = {# call wrapper_model_default_params #}


--
-- LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params();
--
modelQuantizeDefaultParams :: ModelQuantizeParamsPtr -> IO ()
modelQuantizeDefaultParams = {# call wrapper_model_quantize_default_params #}


--
-- LLAMA_API bool llama_mmap_supported();
--
mmapSupported :: IO Bool
mmapSupported = toBool <$> {# call mmap_supported #}


--
-- LLAMA_API bool llama_mlock_supported();
--
mlockSupported :: IO Bool
mlockSupported = toBool <$> {# call mlock_supported #}


--
-- // TODO: not great API - very likely to change
-- // Initialize the llama + ggml backend
-- // If numa is true, use NUMA optimizations
-- // Call once at the start of the program
-- LLAMA_API void llama_init_backend(bool numa);
--
initBackend :: Bool -> IO ()
initBackend = {# call backend_init #} . fromBool


-- // Call once at the end of the program - currently only used for MPI
--
freeBackend :: IO ()
freeBackend = {# call backend_free #}


--
-- LLAMA_API int64_t llama_time_us();
--
timeUs :: IO CLong
timeUs = {# call time_us #}


--
-- LLAMA_API struct llama_model * llama_load_model_from_file(
--                          const char * path_model,
--         struct llama_context_params   params);
--
-- Can't pass structs by value via FFI, so wrote a wrapper:
--
loadModelFromFile :: Ptr CChar -> ModelParamsPtr -> IO Model
loadModelFromFile modelPath modelParamsPtr =
  {# call wrapper_load_model_from_file #} modelPath (castPtr modelParamsPtr)


--
-- LLAMA_API void llama_free_model(struct llama_model * model);
--
freeModel :: Model -> IO ()
freeModel = {# call free_model #}


--
-- LLAMA_API struct llama_context * llama_new_context_with_model(
--                  struct llama_model * model,
--         struct llama_context_params   params);
--

newContextWithModel :: Model -> ContextParamsPtr -> IO (Context)
newContextWithModel model ctxParamsPtr =
  {# call wrapper_new_context_with_model #} model (castPtr ctxParamsPtr)


--
-- // Frees all allocated memory
-- LLAMA_API void llama_free(struct llama_context * ctx);
--

free :: Context -> IO ()
free = {# call free #}


--
-- // Returns 0 on success
-- LLAMA_API int llama_model_quantize(
--         const char * fname_inp,
--         const char * fname_out,
--         const llama_model_quantize_params * params);
--
modelQuantize :: Ptr CChar -> Ptr CChar -> ModelQuantizeParamsPtr -> IO CInt
modelQuantize = {# call model_quantize #}


--
-- LLAMA_API int llama_model_apply_lora_from_file(
--         const struct llama_model * model,
--                   const char * path_lora,
--                   const char * path_base_model,
--                          int   n_threads);
--
modelApplyLoraFromFile :: Model -> Ptr CChar -> CFloat -> Ptr CChar -> CInt -> IO CInt
modelApplyLoraFromFile = {# call model_apply_lora_from_file #}


--
-- // Returns the number of tokens in the KV cache
-- LLAMA_API int llama_get_kv_cache_token_count(const struct llama_context * ctx);
--
getKVCacheTokenCount :: Context -> IO CInt
getKVCacheTokenCount = {# call get_kv_cache_token_count #}


--
-- // Sets the current rng seed.
-- LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed);
--
setRNGSeed :: Context -> Word32 -> IO ()
setRNGSeed ctx = {# call set_rng_seed #} ctx . fromIntegral


--
-- // Returns the maximum size in bytes of the state (rng, logits, embedding
-- // and kv_cache) - will often be smaller after compacting tokens
-- LLAMA_API size_t llama_get_state_size(const struct llama_context * ctx);
--
getStateSize :: Context -> IO CULong
getStateSize = {# call get_state_size #}


--
-- // Copies the state to the specified destination address.
-- // Destination needs to have allocated enough memory.
-- // Returns the number of bytes copied
-- LLAMA_API size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst);
--
copyStateData :: Context -> Ptr CUChar -> IO CULong
copyStateData = {# call copy_state_data #}


--
-- // Set the state reading from the specified address
-- // Returns the number of bytes read
-- LLAMA_API size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src);
--
setStateData :: Context -> Ptr CUChar -> IO CULong
setStateData = {# call set_state_data #}


--
-- // Save/load session file
-- LLAMA_API bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
--
loadSessionFile :: Context -> Ptr CChar -> Ptr Token -> CULong -> Ptr CULong -> IO CUChar
loadSessionFile = {# call load_session_file #}


--
-- LLAMA_API bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
--
saveSessionFile :: Context -> Ptr CChar -> Ptr Token -> CULong -> IO CUChar
saveSessionFile = {# call save_session_file #}


--
-- // Run the llama inference to obtain the logits and probabilities for the next token.
-- // tokens + n_tokens is the provided batch of new tokens to process
-- // n_past is the number of tokens to use from previous eval calls
-- // Returns 0 on success
-- LLAMA_API int llama_eval(
--         struct llama_context * ctx,
--            const llama_token * tokens,
--                          int   n_tokens,
--                          int   n_past,
--                          int   n_threads);
--
eval :: Context -> Ptr CInt -> CInt -> CInt -> IO CInt
eval = {# call eval #}


--
-- // Same as llama_eval, but use float matrix input directly.
-- LLAMA_API int llama_eval_embd(
--         struct llama_context * ctx,
--                  const float * embd,
--                          int   n_tokens,
--                          int   n_past,
--                          int   n_threads);
--
evalEmbd :: Context -> Ptr CFloat -> CInt -> CInt -> IO CInt
evalEmbd = {# call eval_embd #}


--
-- // Export a static computation graph for context of 511 and batch size of 1
-- // NOTE: since this functionality is mostly for debugging and demonstration purposes, we hardcode these
-- //       parameters here to keep things simple
-- // IMPORTANT: do not use for anything else other than debugging and testing!
-- LLAMA_API int llama_eval_export(struct llama_context * ctx, const char * fname);
--
-- evalExport :: Context -> Ptr CChar -> IO CInt
-- evalExport = {# call eval_export #}


--
-- // Convert the provided text into tokens.
-- // The tokens pointer must be large enough to hold the resulting tokens.
-- // Returns the number of tokens on success, no more than n_max_tokens
-- // Returns a negative number on failure - the number of tokens that would have been returned
-- // TODO: not sure if correct
-- LLAMA_API int llama_tokenize(
--         struct llama_context * ctx,
--                   const char * text,
--                  llama_token * tokens,
--                          int   n_max_tokens,
--                         bool   add_bos);
--
tokenize :: Model
                -> Ptr CChar
                -> CInt
                -> Ptr CInt
                -> CInt
                -> CUChar
                -> CUChar
                -> IO CInt
tokenize = {# call tokenize #}

-- LLAMA_API int llama_tokenize_with_model(
--     const struct llama_model * model,
--                   const char * text,
--                  llama_token * tokens,
--                          int   n_max_tokens,
--                         bool   add_bos);
-- tokenizeWithModel :: Model -> Ptr CChar -> Ptr Token -> CInt -> CUChar -> IO CInt
-- tokenizeWithModel = {# call tokenize_with_model #}


--
-- LLAMA_API int llama_n_vocab(const struct llama_context * ctx);
--
nVocab :: Model -> IO CInt
nVocab = {# call n_vocab #}

--
-- LLAMA_API int llama_n_ctx  (const struct llama_context * ctx);
--
nCtx :: Context -> IO CInt
nCtx = {# call n_ctx #}

--
-- LLAMA_API int llama_n_embd (const struct llama_context * ctx);
--
nEmbd :: Model -> IO CInt
nEmbd = {# call n_embd #}


--
-- LLAMA_API int llama_n_vocab_from_model(const struct llama_model * model);
--
-- nVocabFromModel :: Model -> IO CInt
-- nVocabFromModel = {# call n_vocab_from_model #}

--
-- LLAMA_API int llama_n_ctx_from_model  (const struct llama_model * model);
--
-- nCtxFromModel :: Model -> IO CInt
-- nCtxFromModel = {# call n_ctx_from_model #}


--
-- LLAMA_API int llama_n_embd_from_model (const struct llama_model * model);
--
-- nEmbdFromModel :: Model -> IO CInt
-- nEmbdFromModel = {# call n_embd_from_model #}


--
-- // Get the vocabulary as output parameters.
-- // Returns number of results.
-- LLAMA_API int llama_get_vocab(
--         const struct llama_context * ctx,
--                       const char * * strings,
--                              float * scores,
--                                int   capacity);
--
-- getVocab :: Context -> Ptr (Ptr CChar) -> Ptr CFloat -> CInt -> IO CInt
-- getVocab = {# call get_vocab #}


--
-- // Token logits obtained from the last call to llama_eval()
-- // The logits for the last token are stored in the last row
-- // Can be mutated in order to change the probabilities of the next token
-- // Rows: n_tokens
-- // Cols: n_vocab
-- LLAMA_API float * llama_get_logits(struct llama_context * ctx);
--
getLogits :: Context -> IO (Ptr CFloat)
getLogits = {# call llama_get_logits #}


--
-- // Get the embeddings for the input
-- // shape: [n_embd] (1-dimensional)
-- LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
--
getEmbeddings :: Context -> IO (Ptr CFloat)
getEmbeddings = {# call llama_get_embeddings #}


tokenToPiece :: Model -> Token -> IO String
tokenToPiece m t =
    let alloc' size = allocaBytes size $ \buf -> do
            nTokens <- fromIntegral <$> tokenToPiece' m t buf (fromIntegral size)
            if nTokens >= 0
                then Right <$> peekCStringLen (buf, nTokens)
                else pure $ Left (negate nTokens)
     in do
        r <- alloc' 8
        either (\nTokens -> do
            r' <- alloc' nTokens
            pure $ fromRight "??" r'
            ) (pure . id) r

--
-- // Token Id -> String. Uses the vocabulary in the provided context
-- LLAMA_API const char * llama_token_to_str(const struct llama_context * ctx, llama_token token);
--
tokenToPiece' :: Model -> Token -> Ptr CChar -> CInt -> IO CInt
tokenToPiece' = {# call token_to_piece #}


-- LLAMA_API const char * llama_token_to_str_with_model(
--           const struct llama_model * model,
--                        llama_token   token);
-- tokenToStrWithModel :: Model -> Token -> IO (Ptr CChar)
-- tokenToStrWithModel = {# call token_to_str_with_model #}


-- // Special tokens

--
-- LLAMA_API llama_token llama_token_bos();  // beginning-of-sentence
--
tokenBos :: Model -> IO Token
tokenBos = {# call token_bos #}


--
-- LLAMA_API llama_token llama_token_eos();  // end-of-sentence
--
tokenEos :: Model -> IO Token
tokenEos = {# call token_eos #}


--
-- LLAMA_API llama_token llama_token_nl();   // next-line
--
tokenNl :: Model -> IO Token
tokenNl = {# call token_nl #}


-- // Sampling functions

--
-- /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
-- LLAMA_API void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty, float penalty_freq);
--
sampleRepetitionPenalties :: Context -> Ptr TokenDataArray -> Ptr Token -> CULong -> CFloat -> CFloat -> CFloat -> IO ()
sampleRepetitionPenalties = {# call sample_repetition_penalties #}


--
-- /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
-- LLAMA_API void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);
--
-- sampleFrequencyAndPresencePenalties
--   :: Context -> Ptr TokenDataArray -> Ptr Token -> CULong -> CFloat -> CFloat -> IO ()
-- sampleFrequencyAndPresencePenalties = {# call sample_frequency_and_presence_penalties #} 


-- /// @details Apply classifier-free guidance to the logits as described in academic paper "Stay on topic with Classifier-Free Guidance" https://arxiv.org/abs/2306.17806
-- /// @param candidates A vector of `llama_token_data` containing the candidate tokens, the logits must be directly extracted from the original generation context without being sorted.
-- /// @params guidance_ctx A separate context from the same model. Other than a negative prompt at the beginning, it should have all generated and user input tokens copied from the main context.
-- /// @params scale Guidance strength. 1.0f means no guidance. Higher values mean stronger guidance.
-- LLAMA_API void llama_sample_classifier_free_guidance(
--           struct llama_context * ctx,
--         llama_token_data_array * candidates,
--           struct llama_context * guidance_ctx,
--                          float   scale);
sampleClassifierFreeGuidance :: Context -> TokenDataArrayPtr -> Context -> CFloat -> IO ()
sampleClassifierFreeGuidance = {# call sample_classifier_free_guidance #}


--
-- /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
-- LLAMA_API void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates);
--
sampleSoftmax :: Context -> Ptr TokenDataArray -> IO ()
sampleSoftmax = {# call sample_softmax #}


--
-- /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
-- LLAMA_API void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep);
--
sampleTopK :: Context -> Ptr TokenDataArray -> CInt -> CULong -> IO ()
sampleTopK = {# call sample_top_k #}


--
-- /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
-- LLAMA_API void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
--
sampleTopP :: Context -> Ptr TokenDataArray -> CFloat -> CULong -> IO ()
sampleTopP = {# call sample_top_p #}


--
-- /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
-- LLAMA_API void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep);
--
sampleTailFree :: Context -> Ptr TokenDataArray -> CFloat -> CULong -> IO ()
sampleTailFree = {# call sample_tail_free #}


--
-- /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
-- LLAMA_API void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
--
sampleTypical :: Context -> Ptr TokenDataArray -> CFloat -> CULong -> IO ()
sampleTypical = {# call sample_typical #}


--
-- LLAMA_API void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates, float temp);
--
sampleTemperature :: Context -> Ptr TokenDataArray -> CFloat -> IO ()
sampleTemperature = {# call sample_temperature #}


--
-- /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
-- /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
-- /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
-- /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
-- /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
-- /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
-- LLAMA_API llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu);
--
sampleTokenMirostat :: Context -> Ptr TokenDataArray -> CFloat -> CFloat -> CInt -> Ptr CFloat -> IO Token
sampleTokenMirostat = {# call sample_token_mirostat #}


--
-- /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
-- /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
-- /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
-- /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
-- /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
-- LLAMA_API llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu);
--
sampleTokenMirostatV2 :: Context -> Ptr TokenDataArray -> CFloat -> CFloat -> Ptr CFloat -> IO Token
sampleTokenMirostatV2 = {#call sample_token_mirostat_v2 #}


--
-- /// @details Selects the token with the highest probability.
-- LLAMA_API llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates);
--
sampleTokenGreedy :: Context -> Ptr TokenDataArray -> IO Token
sampleTokenGreedy = {# call sample_token_greedy #}


--
-- /// @details Randomly selects a token from the candidates based on their probabilities.
-- LLAMA_API llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates);
--
sampleToken :: Context -> Ptr TokenDataArray -> IO Token
sampleToken = {# call sample_token #}


--
-- // Performance information
--

--
--     // performance timing information
--     struct llama_timings {
--         double t_start_ms;
--         double t_end_ms;
--         double t_load_ms;
--         double t_sample_ms;
--         double t_p_eval_ms;
--         double t_eval_ms;
--
--         int32_t n_sample;
--         int32_t n_p_eval;
--         int32_t n_eval;
--     };
--
data Timings = Timings
  { _tStartMs :: CDouble
  , _tEndMs :: CDouble
  , _tLoadMs :: CDouble
  , _tSampleMs :: CDouble
  , _tPEvalMs :: CDouble
  , _tEvalMs :: CDouble
  , _nSample :: Int
  , _nPEval :: Int
  , _nEval :: Int
  }
  deriving (Eq, Show)

{# pointer *timings as TimingsPtr -> Timings #}

instance Storable Timings where
  sizeOf _ = {# sizeof timings #}
  alignment _ = {# alignof timings #}
  peek p = Timings
    <$> {# get timings->t_start_ms #} p
    <*> {# get timings->t_end_ms #} p
    <*> {# get timings->t_load_ms #} p
    <*> {# get timings->t_sample_ms #} p
    <*> {# get timings->t_p_eval_ms #} p
    <*> {# get timings->t_eval_ms #} p
    <*> (fromIntegral <$> {# get timings->n_sample #} p)
    <*> (fromIntegral <$> {# get timings->n_p_eval #} p)
    <*> (fromIntegral <$> {# get timings->n_eval #} p)
  poke p (Timings _tStartMs _tEndMs _tLoadMs _tSampleMs _tPEvalMs _tEvalMs _nSample _nPEval _nEval) = do
    {# set timings->t_start_ms #} p _tStartMs
    {# set timings->t_end_ms #} p _tEndMs
    {# set timings->t_load_ms #} p _tLoadMs
    {# set timings->t_sample_ms #} p _tSampleMs
    {# set timings->t_p_eval_ms #} p _tPEvalMs
    {# set timings->t_eval_ms #} p _tEvalMs
    {# set timings->n_sample #} p $ fromIntegral _nSample
    {# set timings->n_p_eval #} p $ fromIntegral _nPEval
    {# set timings->n_eval #} p $ fromIntegral _nEval


--
-- LLAMA_API struct llama_timings llama_get_timings(struct llama_context * ctx);
--
getTimings :: Context -> Ptr Timings -> IO ()
getTimings = {# call wrapper_get_timings #}

--
-- LLAMA_API void llama_print_timings(struct llama_context * ctx);
--
printTimings :: Context -> IO ()
printTimings = {# call print_timings #}


--
-- LLAMA_API void llama_reset_timings(struct llama_context * ctx);
--
resetTimings :: Context -> IO ()
resetTimings = {# call reset_timings #}


--
-- // Print system information
-- LLAMA_API const char * llama_print_system_info(void);
--
printSystemInfo :: IO (Ptr CChar)
printSystemInfo = {# call print_system_info #}
