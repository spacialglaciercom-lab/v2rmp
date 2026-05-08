use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::{PaddingParams, Tokenizer};

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let api = Api::new()?;
        let repo = api.repo(Repo::model("BAAI/bge-small-en-v1.5".to_string()));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(anyhow::Error::msg)
            .context("Failed to load tokenizer")?;

        let tensors = candle_core::safetensors::load(&weights_filename, &device)?;
        let vb = VarBuilder::from_tensors(tensors, DTYPE, &device);
        let model = BertModel::load(vb, &config)?;

        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let tokens = self.tokenizer.encode_batch(texts, true)
            .map_err(anyhow::Error::msg)
            .context("Failed to tokenize batch")?;

        let token_ids = tokens.iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &self.device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;
        
        // Mean pooling
        let (_n_batch, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        
        // Normalize
        let norm = embeddings.sqr()?.sum_keepdim(1)?.sqrt()?;
        let embeddings = embeddings.broadcast_div(&norm)?;
        
        let embeddings = embeddings.to_vec2::<f32>()?;
        Ok(embeddings)
    }
}

pub fn run_embed(texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
    let mut embedder = Embedder::new()?;
    embedder.embed(texts)
}
