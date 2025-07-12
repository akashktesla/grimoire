#![allow(warnings)]
use std::collections::HashSet;

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

