
#include <llama.h>
#include "wrapper.h"

struct llama_model * wrapper_load_model_from_file
( const char * path_model, struct llama_context_params * params) {
  return llama_load_model_from_file(path_model, *params);
}

void wrapper_context_default_params(struct llama_context_params * default_params) {
  *default_params = llama_context_default_params();
}

void wrapper_model_quantize_default_params(struct llama_model_quantize_params * default_params) {
  *default_params = llama_model_quantize_default_params();
}

void wrapper_get_timings(struct llama_context * ctx, struct llama_timings * timings) {
  *timings = llama_get_timings(ctx);
}
