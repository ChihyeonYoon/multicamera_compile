video_ch0_path="compile_materials/thelive/W.mp4"
video_ch1_path="compile_materials/thelive/C.mp4"
video_ch2_path="compile_materials/thelive/D.mp4"
video_chMC_path="compile_materials/thelive/MC_48-78.mp4"

start_time=0
end_time=10

sample_name="sample_thelive"

inference_result_dict_path="compiled_sample/$sample_name.json"
output_video_path="compiled_sample/$sample_name.mp4" 
# output video filename will be {$sample_name}.mp4

model_path="speaking_detection_model_weight.pth"

sleep 1s

python video_channel_inference_the_live.py --video_ch0_path $video_ch0_path \
 --video_ch1_path $video_ch1_path \
 --video_ch2_path $video_ch2_path \
 --video_chMC_path $video_chMC_path \
 --start_time $start_time \
 --end_time $end_time \
 --inference_result_dict_path $inference_result_dict_path \
 --model_path $model_path \

sleep 3s

python video_channel_compile_the_live.py --video_ch0_path $video_ch0_path \
 --video_ch1_path $video_ch1_path \
 --video_ch2_path $video_ch2_path \
 --video_chMC_path $video_chMC_path \
 --inference_result_dict_path $inference_result_dict_path \
 --output_video_path $output_video_path \