python rrs.py \
--linkdata_pos "../../../Data/Processed_Data/Wine/D.pkl" \
--linkdata_neg "../../../Data/Processed_Data/Wine/D_C.pkl" \
--flic "../../../Data/Processed_Data/Wine/FLIC_model_attdim100_nsteps1000.pkl" \
--reviews "../../../Data/Processed_Data/Wine/reviews_embedded_256.pkl" \
--save_model "../../../Data/Processed_Data/Wine/Model/" \
--enc_num_hidden_units 100 \
--enc_num_hidden_layers 1 \
--gen_num_hidden_units 100 \
--gen_num_hidden_layers 1 \
--max_epochs 200 \
--num_samples 10 \
--num_target_sentences 5 \
--category wine \
--adaptive_lrs \
--regularizer 0.005 \
--measure_timing \
--model_type DPP \
--context train_context \

