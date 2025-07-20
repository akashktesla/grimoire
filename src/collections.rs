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

    pub fn insert_node(&mut self, node_id:&usize, embedding:&Embedding){
    if !self.inserted_ids.insert(*node_id) {
        return; // Already inserted
           }
        self.inserted_ids.insert(*node_id);
        let core_similarity = cosine_similarity(&self.core_node.embedding.embedding,&embedding.embedding);
        if self.nodes.len() >= self.k{ //aldready full
            //sum of similarity
            let mut sos = 0.;
            for i in &self.nodes{
                let similarity  = cosine_similarity(&embedding.embedding,&i.embedding.embedding);
                sos += similarity;//minimize this
            }
            let  diversity_score = self.k1*(self.nodes.len() as f32-sos)+self.k2*core_similarity;
            self.nodes.sort_by(|a,b|b.diversity_score.partial_cmp(&a.diversity_score).unwrap());
            self.nodes.pop();
        }
        else{
            let mut sos = 0.;
            for i in &self.nodes{
                let similarity  = cosine_similarity(&embedding.embedding,&i.embedding.embedding);
                sos += similarity;//minimize this
            }
            let  diversity_score = self.k1*(self.nodes.len() as f32-sos)+self.k2*core_similarity;
            self.nodes.push(KDNode::new(*node_id,embedding.clone(),diversity_score));
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
