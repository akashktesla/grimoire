#![allow(warnings)]
pub fn main(){
    let embedding = vec![0.000000002,0.0032,0.00043,0.14];
    let num = 0.0000334234001; 
    let exponent = get_exponent(num);
    println!("Exponent: {:?}",exponent);
    // let chunk_number = generate_metadata(embedding,10);
    // println!("chunk_number: {:?}",chunk_number);
    // let chunk_number_1 = vec![4,6,7,3];
    // let chunk_number_2 = vec![4,5,8,3];
    // let diff = calculate_difference(chunk_number_1,chunk_number_2);
    // println!("diff: {:?}",diff);

}

fn get_exponent(x: f32) -> (f32, i32) {
    if x == 0.0 {
        return (0.0, 0);
    }

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    let bits = abs_x.to_bits();
    let exp2 = ((bits >> 23) & 0xFF) as i32 - 127;
    let mut exp10 = (exp2 as f32 * 0.30103).floor() as i32;

    let mut mantissa = abs_x / 10f32.powi(exp10);

    // Normalize to [1.0, 10.0)
    if mantissa >= 10.0 {
        mantissa /= 10.0;
        exp10 += 1;
    } else if mantissa < 1.0 {
        mantissa *= 10.0;
        exp10 -= 1;
    }

    (sign * mantissa, exp10)
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

