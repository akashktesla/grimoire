#![allow(warnings)]
use std::collections::HashSet;
use crate::grimoire::Embedding;
use crate::hellindex::cosine_similarity;

pub fn main(){
    let mut array = KSArray::new(3); 
    array.insert_node(&0,&0.5);
    array.insert_node(&0,&0.70);
    array.insert_node(&0,&0.71);
    array.insert_node(&0,&0.6);
    array.insert_node(&0,&0.2);
    array.insert_node(&0,&0.4);
    println!("array: {:?}",array);
}

#[derive(Debug, Clone)]
pub struct KSArray{
    pub nodes:Vec<KSNode>,
    pub k:usize,
    inserted_ids: HashSet<usize>, 
}

impl KSArray{

    pub fn new(k:usize)->KSArray{
        let nodes = Vec::new();
        return KSArray { 
            nodes , 
            k ,
            inserted_ids:HashSet::new(),
        };
    }

    pub fn insert_node(&mut self, node_id:&usize, similarity:&f32){
        if self.inserted_ids.contains(node_id){
            return;
        }
        self.inserted_ids.insert(*node_id);
        if self.nodes.len() >= self.k{ //aldready full
            if *similarity < self.nodes.last().unwrap().similarity { //optimization 
                return; //break the function
            }
            self.nodes.push(KSNode::new(*node_id,*similarity));
            self.nodes.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            self.nodes.pop();
        }
        else{
            self.nodes.push(KSNode::new(*node_id,*similarity));
            self.nodes.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        }
    }
} 

//KSimilar array
#[derive(Debug, Clone)]
pub struct KSNode{
    pub node_id:usize,
    pub similarity:f32,
}

impl KSNode{
    pub fn new(node_id:usize, similarity:f32)->KSNode{
        return KSNode{
            node_id,
            similarity

        }
    }
}

//K Diverse Array
#[derive(Debug, Clone)]
pub struct KDArray{
    pub nodes:Vec<KDNode>,
    pub core_node:KDNode,
    pub k:usize,
    inserted_ids: HashSet<usize>, 
    k1:f32,
    k2:f32,
}

impl KDArray{

    pub fn new(node_id:&usize,embedding:&Embedding,k:usize,k1:f32,k2:f32)->KDArray{
        let nodes = Vec::new();
        let core_node = KDNode::new(*node_id, embedding.clone(), 0.);
        return KDArray { 
            nodes , 
            k,
            core_node,
            inserted_ids:HashSet::new(),
            k1,
            k2
        };
    }
    pub fn insert_node(&mut self, node_id: &usize, embedding: &Embedding) {
        // Skip if already inserted
        if !self.inserted_ids.insert(*node_id) {
            return;
        }

        // Compute similarity to core node once
        let core_similarity = cosine_similarity(&self.core_node.embedding.embedding, &embedding.embedding);

        // Precompute sum of similarities to existing nodes
        let sos: f32 = self.nodes.iter()
            .map(|n| cosine_similarity(&embedding.embedding, &n.embedding.embedding))
            .sum();

        let diversity_score = self.k1 * ((self.nodes.len() as f32) - sos) + self.k2 * core_similarity;
        let new_node = KDNode::new(*node_id, embedding.clone(), diversity_score);

        if self.nodes.len() < self.k {
            // Simply push if under capacity
            self.nodes.push(new_node);
            return;
        }
        // Already full: Find the *worst* node (min score), and replace if new one is better
        if let Some((worst_idx, worst_score)) = self.nodes.iter()
            .enumerate()
                .min_by(|(_, a), (_, b)| a.diversity_score.partial_cmp(&b.diversity_score).unwrap()) 
        {
            if diversity_score > worst_score.diversity_score {
                self.nodes[worst_idx] = new_node;
            }
        }
    }

} 


#[derive(Debug, Clone)]
pub struct KDNode{
    pub node_id:usize,
    pub embedding:Embedding,
    pub diversity_score:f32,
}

impl KDNode{
    pub fn new(node_id:usize, embedding:Embedding,diversity_score:f32)->KDNode{
        return KDNode{
            node_id,
            embedding,
            diversity_score,
        }
    }
}
