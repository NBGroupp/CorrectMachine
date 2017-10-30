export DATA_PATH=./data

export VOCAB_SOURCE=${DATA_PATH}/vocab
export VOCAB_TARGET=${DATA_PATH}/vocab
export TRAIN_SOURCES=${DATA_PATH}/error0_train
export TRAIN_TARGETS=${DATA_PATH}/target_train
export DEV_SOURCES=${DATA_PATH}/error0_dev
export DEV_TARGETS=${DATA_PATH}/target_dev

export DEV_TARGETS_REF=${DATA_PATH}/target_dev
export TRAIN_STEPS=1000000
export MODEL_DIR=${TMPDIR:-.}/checkpoints
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
