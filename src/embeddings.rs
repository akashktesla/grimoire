// #![allow(warnings)]
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };

pub fn generate_embeddings_vec(payload:Vec<String>)->Vec<Vec<f32>>{
    let model:SentenceEmbeddingsModel = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2). 
        create_model() 
        .expect("Couldn't Load the embedding model");
    return model.encode(&payload).expect("Failed to encode the string");
}

pub fn generate_embeddings_string(payload:&String)->Vec<f32>{
    let model:SentenceEmbeddingsModel = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2). 
        create_model() 
        .expect("Couldn't Load the embedding model");
    return model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
}
