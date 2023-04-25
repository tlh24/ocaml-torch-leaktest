open Torch

let image_count = 4*2048
let image_res = 30

let foi = float_of_int

(* helper routine to print GPU memory usage *)
let run_nvidiasmi () = 
	let print_chan channel =
		let rec loop () =
			print_endline (input_line channel);
			loop ()
		in
		try loop ()
		with End_of_file -> close_in channel
	in
	let (ocaml_stdout, ocaml_stdin, ocaml_stderr) = 
		Unix.open_process_full "nvidia-smi | grep leaktest" [||] in
	close_out ocaml_stdin;
	print_chan ocaml_stdout;
	print_chan ocaml_stderr


(* algorithm A, more obvious *)
let image_dist_a dbf img =
	let d = Tensor.( (dbf - img) ) in
	let d2 = Tensor.einsum ~equation:"ijk, ijk -> i" [d;d] ~path:None in
	let mindex = Tensor.argmin d2 ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d2 mindex |> Tensor.float_value in
	dist,mindex

(* algorithm B, more efficient *)
let image_dist_b dbf img = 
	let _dbfn,h,_ = Tensor.shape3_exn dbf in
	let b = Tensor.expand img ~implicit:true ~size:[1;h;h] in
	let d = Tensor.(sum_dim_intlist (square(dbf - b)) 
			~dim:(Some [1;2]) ~keepdim:false ~dtype:(T Float) ) in
	let mindex = Tensor.argmin d ~dim:None ~keepdim:true 
		|> Tensor.int_value in
	let dist = Tensor.get d mindex |> Tensor.float_value in
	dist,mindex


let () = 
	Printf.printf "cuda available: %b\n%!" (Cuda.is_available ());
(* 	let device = Torch.Device.Cpu in *)
	let device = Torch.Device.cuda_if_available () in
	let dbf = Tensor.( (ones [image_count; image_res; image_res] ) * (f (-1.0)))
		|> Tensor.to_device ~device in
	let siz = image_count * image_res * image_res * 4 in
	Printf.printf "database size: %d bytes %f MB\n" siz ((foi siz) /. 1e6); 
	Printf.printf "[00]"; run_nvidiasmi (); 
	for i = 1 to 30 do (
		(* generate a random image *)
		Caml.Gc.full_major();
		let img = Tensor.(randn [image_res; image_res] ) 
			|> Tensor.to_device ~device in
		
		(* you can run either MSE calculation, both leak memory 
			Demo running both 
			Running image_dist_* within Tensor.no_grad does not change *)
			
		let dist_a,mindex_a = image_dist_a dbf img in
		let dist_b,mindex_b = image_dist_b dbf img in
		let df = (abs_float dist_a -. dist_b) in
		assert (df < 0.01); 
		assert (mindex_a = mindex_b); 
		
		Printf.printf "[%02d]" i; run_nvidiasmi ()
	) done
