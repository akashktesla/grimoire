#![allow(warnings)]
pub fn main(){
    let embedding = vec![0.000000002,0.0032,0.00043,0.14];
    let normalized_embedding = normalization(embedding,10);
    println!("normalized_embedding: {:?}",normalized_embedding);
    // let chunk_number = generate_metadata(embedding,10);
    // println!("chunk_number: {:?}",chunk_number);
    // let chunk_number_1 = vec![4,6,7,3];
    // let chunk_number_2 = vec![4,5,8,3];
    // let diff = calculate_difference(chunk_number_1,chunk_number_2);
    // println!("diff: {:?}",diff);

}


fn normalization(mut embedding: Vec<f32>,chunk_size:i32 ){
    embedding.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("sorted: {:?}",embedding);

}


pub fn generate_metadata(embedding: &Vec<f32>,chunk_size:&i32)->(Vec<i32>,i32){
   let chunk_number:Vec<i32> = embedding
       .into_iter()
       .map(|i| (i*(*chunk_size as f32)) as i32)
       .collect();

    let rank = chunk_number
        .iter()
        .sum();

    return (chunk_number,rank)
}

pub fn calculate_difference(chunk_number_1:Vec<i32>,chunk_number_2:Vec<i32>)->i32{
    return chunk_number_1
        .into_iter()
        .zip(chunk_number_2)
        .map(|(a,b)|(a-b).abs())
        .sum();
}

