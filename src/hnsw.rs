#![allow(warnings)]
use crate::grimoire::Embedding;
use crate::hellindex::cosine_similarity;
use std::{collections::HashMap, hash::Hash};
use rust_bert::pipelines::sentence_embeddings::{ SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType, SentenceEmbeddingsModel };
use rand::Rng;
use std::fmt;
use std::fmt::Debug;

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
    let max_neighbours:usize = 5;
    let eq_construction :usize = 200;
    let eq_search:usize = 70;
    let mut engine:HnswEngine = HnswEngine::new(max_level, embedding_model_path, level_probability, eq_construction, eq_search,max_neighbours); 
    engine.load(sentences);
    println!("{:?}",engine);
    
}

type NodeId = usize;
    
#[derive(Debug, Clone)]
struct Neighbour {
    node_id: NodeId,
    similarity: f32,
}

impl Neighbour{
    fn new(node_id:NodeId,similarity:f32)->Neighbour{
        return Neighbour{
            node_id,
            similarity
        }

    }
}


#[derive(Clone, Debug)]
struct HnswNode{
    id:NodeId,
    embedding: Embedding, 
    level:usize, //level of the node
    neighbours: HashMap<usize, Vec<Neighbour>> // level - node_id
}

impl HnswNode{
    fn new(id:NodeId,embedding:Embedding,level:usize,neighbours:HashMap<usize,Vec<Neighbour>>)->HnswNode{
        return HnswNode{
            id,
            embedding,
            level,
            neighbours
        }
    }

    fn new_empty()->HnswNode{
        return HnswNode{
            id:0,
            embedding:Embedding::new_empty(),
            level:0,
            neighbours:HashMap::new()
        }

    }

    fn insert_neighbour(&mut self, level:&usize, node_id:&NodeId, similarity:&f32, max_neighbours:&usize){
        match self.neighbours.get_mut(&level){
            Some(neighbours)=>{
                if &(neighbours.len() as usize) > max_neighbours { //aldready full
                    if *similarity < neighbours.last().unwrap().similarity { //optimization 
                        return; //break the function
                    }
                    neighbours.push(Neighbour::new(*node_id,*similarity));
                    neighbours.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
                    neighbours.pop();
                }
                else{
                    neighbours.push(Neighbour::new(*node_id,*similarity));
                    neighbours.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
                }
            },
            None=>{
                self.neighbours.insert(*level,vec![Neighbour::new(*level,*similarity)]);
            }
        }
    }
}


struct HnswEngine{
    entry_point: HnswNode, //dynamically updated
    max_level:usize, // for tracking
    embedding_model:SentenceEmbeddingsModel,
    embedding_model_path:String,
    nodes: HashMap<NodeId,HnswNode>, //global pool of nodes
    level_nodes: HashMap<usize,Vec<NodeId>>,
    current_node_id: NodeId,
    level_probability:f64,
    max_neighbours:usize,
    eq_construction:usize,
    eq_search:usize,
}

impl Debug for HnswEngine{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HnswEngine: 
            entry_point: {:?}, 
            max_level: {},
            embedding_model_path: {},
            nodes: {:#?},
            level_nodes: {:?},
            current_node_id: {},
            level_probability: {},
            max_neighbours: {},
            eq_construction: {},
            eq_search: {}
            ",
            self.entry_point,
            self.max_level,
            self.embedding_model_path,
            self.nodes,
            self.level_nodes,
            self.current_node_id,
            self.level_probability,
            self.max_neighbours,
            self.eq_construction,
            self.eq_search)
    }

}



impl HnswEngine{
    fn new(max_level:usize, embedding_model_path:String,
        level_probability:f64, eq_construction:usize, eq_search:usize, max_neighbours:usize) -> HnswEngine{


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
            level_nodes:HashMap::new(),
            current_node_id:0,
            level_probability,
            max_neighbours,
            eq_construction,
            eq_search,
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
            let current_node_id = self.current_node_id.clone();
            self.nodes.insert(current_node_id,  node);

            //level nodes relationship 
            match self.level_nodes.get_mut(&level){
                Some(node_list)=>{
                    node_list.push(current_node_id);
                }
                None=>{
                    self.level_nodes.insert(level,vec![current_node_id]);
                }
            }
            self.update_neighbours_greedy(&current_node_id,level);
            self.current_node_id = self.current_node_id+1
        }
    }

    fn update_neighbours_greedy(&mut self,node_id:&NodeId,level:usize){
        let embedding = &self.nodes.get(node_id).unwrap().embedding.clone();

        let mut similarities_vec= Vec::new();
        let node_list  = self.level_nodes.get(&level).unwrap();
        //TODO: to implement a greedy approach when brute force > eq_construction
        for i in node_list{ 
            let sec_embedding = self.nodes.get(i).unwrap().embedding.embedding.clone();
            let similarity = cosine_similarity(&embedding.embedding, &sec_embedding);
            similarities_vec.push((*i,similarity));
        }
        //sorting by most similarity
        similarities_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        //inserting neighbors after sorting
        let b = 5;
        let max_neighbours = self.max_neighbours as usize;
        let node = self.nodes.get_mut(node_id).unwrap();
        for i in &similarities_vec[0..self.max_neighbours.min(similarities_vec.len())]{ 
            node.insert_neighbour(&level,&i.0,&i.1,&self.max_neighbours); 
        }
        //updating neighbours
        for i in &similarities_vec{ 
            let neighbour_node = self.nodes.get_mut(&i.0).unwrap();
            neighbour_node.insert_neighbour(&level,&i.0,&i.1,&self.max_neighbours); 
        }
    }

    fn traverse(user_query:String,){

    }

    fn generate_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen_range(0.0..1.0);
        let scale = 1.0 / self.level_probability.ln(); // scale = 1 / ln(1 / prob)
        let level = (-r.ln() * scale).floor() as usize;
        level.min(self.max_level) // cap to max_level
    }

    fn generate_embeddings_string(&self,payload:&String)->Embedding{
        let embedding = self.embedding_model.encode(&vec![payload]).expect("Failed to encode the string")[0].clone();
        return Embedding::new(payload.clone(),embedding);
      }

}
