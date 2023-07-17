
#include "llama.h"
#include "wrapper.h"

struct llama_model * wrapper_load_model_from_file(
  const char * path_model,
  struct llama_context_params * params) {

  return llama_load_model_from_file(path_model, *params);

}
