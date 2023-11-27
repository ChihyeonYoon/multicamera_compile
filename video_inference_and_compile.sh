video_ch0_path="compile_materials/opentalk/camera1_synced.mp4"
video_ch1_path="compile_materials/opentalk/camera2_synced.mp4"
video_ch2_path="compile_materials/opentalk/camera3_synced.mp4"

start_time=0
end_time=10

sample_name="sample"

inference_result_dict_path="compiled_sample/$sample_name.json"
output_video_path="compiled_sample/$sample_name.mp4" 
# output video filename will be {$sample_name}_version1.mp4, {$sample_name}_version2.mp4

model_path="speaking_detection_model_weight.pth"

sleep 1s

python video_channel_inference.py --video_ch0_path $video_ch0_path \
 --video_ch1_path $video_ch1_path \
 --video_ch2_path $video_ch2_path \
 --start_time $start_time \
 --end_time $end_time \
 --inference_result_dict_path $inference_result_dict_path \
 --model_path $model_path \

sleep 3s

python video_channel_compile.py --video_ch0_path $video_ch0_path \
 --video_ch1_path $video_ch1_path \
 --video_ch2_path $video_ch2_path \
 --inference_result_dict_path $inference_result_dict_path \
 --output_video_path $output_video_path \