#![allow(warnings)]
use bincode::{config, Decode, Encode};
use std::fs::{write, read};
use crate::embeddings::{generate_embeddings_string,generate_embeddings_vec};

pub fn main() {
    let path = "/home/akash/projects/grimoire/src/test.grm".to_string();
    let payload = vec!["Akash loves cooking".to_string(),"Akash does all night coding".to_string()];
    let mut vdb = Grimoire::new(path,payload);
    // vdb.save_db();
    // vdb.load_db();
    vdb.insert_string("Akash also loves fucking with others".to_string());
    println!("vdb: {:#?}",vdb);
}

#[derive(Encode, Decode, PartialEq, Debug)]
struct Grimoire{
    path: String,
    db: Vec<(String,Vec<f32>)>
}

impl Grimoire{
    fn new(path:String,mut payload:Vec<String>)->Self{
        let embeddings = generate_embeddings_vec(payload.clone());
        let db = payload.into_iter().zip(embeddings).collect();
        return Grimoire{
            path,
            db
        }
    }
    
    fn save_db(&self){
        let config = config::standard();
        let encoded = bincode::encode_to_vec(self,config).unwrap();
        write(self.path.clone(),&encoded)
            .expect(&format!("Unablel to write the file on path: {}",self.path));
    }

    fn load_db(&mut self){
        let contents = read(self.path.clone()).unwrap();
        let config = config::standard();
        let (loaded,_):(Grimoire,usize) = bincode::decode_from_slice(&contents,config).unwrap();
        *self = loaded;
        println!("contents: {:?}", self);
    }

    fn insert_string(&mut self,payload:String){
        self.db.push((payload.clone(),generate_embeddings_string(payload)));
    }

}
