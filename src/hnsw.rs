#![allow(warnings)]
use crate::grimoire::Embedding;
use crate::hellindex::cosine_similarity;
use std::collections::HashMap;
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };
use rand::Rng;

pub fn main(){

    let sentences = vec![
        String::from("Akash is a god"),
        String::from("Life is all about ups and downs"),
        String::from("I like rust language it has memory safety"),
        String::from("Rust is a good language"),
        String::from("Akash is a great guy"),
        String::from("God doesn't exist"),
    ];
    let max_level = 5;
     
    let embedding_model_path:String = "/home/akash/.models/all-MiniLM-L6-v2".to_string();
    let level_probability:f64 = 0.4; //rounded from 0,36;
    let eq_construction :u32 = 200;
    let eq_search:u32 = 70;
    let mut engine:HnswEngine = HnswEngine::new(max_level, embedding_model_path, level_probability, eq_construction, eq_search); 
    engine.load(sentences);
    
}

type NodeId = u32;
#[derive(Clone, Debug)]
struct HnswNode{
    id:NodeId,
    embedding: Embedding, 
    level:u32, //level of the node
    levels: HashMap<u32, Vec<NodeId>> // level - node_id
}

impl HnswNode{
    fn new(id:NodeId,embedding:Embedding,level:u32,levels:HashMap<u32,Vec<NodeId>>)->HnswNode{
        return HnswNode{
            id,
            embedding,
            level,
            levels
        }
    }

    fn new_empty()->HnswNode{
        return HnswNode{
            id:0,
            embedding:Embedding::new_empty(),
            level:0,
            levels:HashMap::new()
        }

    }
}

struct HnswEngine{
    entry_point: HnswNode, //dynamically updated
    max_level:u32, // for tracking
    embedding_model:SentenceEmbeddingsModel,
    embedding_model_path:String,
    nodes: HashMap<NodeId,HnswNode>, //global pool of nodes
    current_node_id: NodeId,
    level_probability:f64,
    eq_construction:u32,
    eq_search:u32
}

impl HnswEngine{
    fn new(max_level:u32, embedding_model_path:String,
        level_probability:f64, eq_construction:u32, eq_search:u32) -> HnswEngine{
        let embedding_model = SentenceEmbeddingsBuilder
            ::local(&embedding_model_path)
            .create_model()
            .expect("couldn't create the model");

        let entry_point = HnswNode::new_empty();
        return HnswEngine{
            entry_point,
            max_level,
            embedding_model,
            embedding_model_path,
            nodes:HashMap::new(),
            current_node_id:0,
            level_probability,
            eq_construction,
            eq_search

        }
    }

    fn load(&mut self, chunks:Vec<String>){
        for i in chunks{
            let embedding = self.generate_embeddings_string(&i);
            let level = self.generate_level();
            let node = HnswNode::new(self.current_node_id,embedding,level,HashMap::new());
            if level > self.max_level {
                self.entry_point = node.clone();
            }
            //TODO Greedy search here to find neighbors
            self.find_neighbours_greedy(&node.embedding);
            self.nodes.insert(self.current_node_id,  node);
        }
    }

    fn find_neighbours_greedy(&self,user_query:&Embedding){
        for i in self.nodes.values(){
            let similarity = cosine_similarity(&user_query.embedding , &i.embedding.embedding);
            println!("similarity: {}",similarity);
        }
    }

    fn traverse(user_query:String,){

    }

    fn generate_level(&self) -> u32 {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        let scale = 1.0 / self.level_probability.ln(); // scale = 1 / ln(1 / prob)
        let level = (-r.ln() * scale).floor() as u32;
        level.min(self.max_level) // cap to max_level
    }

    fn generate_embeddings_string(&self,payload:&String)->Embedding{
        let embedding = self.embedding_model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
        return Embedding::new(payload.clone(),embedding);
      }

}
