import json
import cv2
import time
import os
from moviepy.editor import AudioFileClip, VideoFileClip
from collections import Counter
import argparse

def frame_number_to_hhmmss(frame_number, frames_per_second=30):
    total_seconds = frame_number / frames_per_second
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_ch0_path', type=str, default=f'compile_materials/thelive/W.mp4',
                        help='video ch0 path') # wide 비디오
    parser.add_argument('--video_ch1_path', type=str, default=f'compile_materials/thelive/C.mp4',
                        help='video ch1 path') # C 비디오
    parser.add_argument('--video_ch2_path', type=str, default=f'compile_materials/thelive/D.mp4',
                        help='video ch2 path') # D 비디오
    parser.add_argument('--video_chMC_path', type=str, default=f'compile_materials/thelive/MC.mp4',
                        help='video chMC path') # MC 비디오
    
    parser.add_argument('--inference_result_dict_path', type=str, default=f'compiled_sample/sample_thelive.json',
                        help='inference output path')
    parser.add_argument('--output_video_path', type=str, default=f'compiled_samples/sample.mp4',
                        help='output video path') 
    args = parser.parse_args()
    
    result_dict = json.load(open(args.inference_result_dict_path))
    frames = [*result_dict.values()]
    selected_channels = [result_dict[key]['selected_channel'] for key in result_dict.keys()]

    # ================== selected channel adjusting ==================
    adjusted_channels = []
    for i in range(0, len(selected_channels), 15):
        batch = selected_channels[i:i+15]
        # print(batch)
        mcv = Counter(batch).most_common(1)[0][0]
        adjusted_channels = adjusted_channels+([mcv]*len(batch))
    
    video_ch0 = cv2.VideoCapture(args.video_ch0_path) # wide
    video_ch1 = cv2.VideoCapture(args.video_ch1_path) # 여자
    video_ch2 = cv2.VideoCapture(args.video_ch2_path) # 남자
    video_chMC = cv2.VideoCapture(args.video_chMC_path) # MC 비디오
    audio = AudioFileClip(args.video_ch0_path)

    start_frame = frames[0]['frame_n'] 
    end_frame = frames[-1]['frame_n']
    start_time = frame_number_to_hhmmss(start_frame)
    end_time = frame_number_to_hhmmss(end_frame)
    print('start_time: ', start_time)
    print('end_time: ', end_time)

    tmp_path = 'tmp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(tmp_path, fourcc, 30.0, (1920, 1080))

    r_start_time = time.time()
    min_total_frame = min(video_ch1.get(cv2.CAP_PROP_FRAME_COUNT), video_ch2.get(cv2.CAP_PROP_FRAME_COUNT), video_ch0.get(cv2.CAP_PROP_FRAME_COUNT))
    print('min_total_frame: ', min_total_frame)
    while (video_ch1.isOpened and video_ch2.isOpened and video_ch0.isOpened):

        ret_ch0, frame_ch0 = video_ch0.read()
        ret_ch1, frame_ch1 = video_ch1.read()
        ret_ch2, frame_ch2 = video_ch2.read()

        ret_chMC, frame_chMC = video_chMC.read() # MC프레임
        
        frame_n = int(video_ch0.get(cv2.CAP_PROP_POS_FRAMES))

        if str(frame_n) in result_dict.keys():
            print("#Frame: {}/{} -------------------------".format(frame_n, end_frame))
            
            selected_channel = adjusted_channels[frame_n - start_frame]
            print('\tselected_channel: ', selected_channel)

            if selected_channel == 0:
                output_video.write(frame_ch0)
            elif selected_channel == 1:
                output_video.write(frame_ch1)
            elif selected_channel == 2:
                output_video.write(frame_ch2)
            elif selected_channel == 3:
                output_video.write(frame_chMC)       
        else:
            continue
    
        if frame_n >= min_total_frame or frame_n >= end_frame:
            break
    output_video.release()
    
    time.sleep(3)

    output_video_with_audio = VideoFileClip(tmp_path)
    os.remove(tmp_path)
    time.sleep(3)
    audio = audio.subclip(t_start = start_time, t_end = end_time)
    output_video_with_audio = output_video_with_audio.set_audio(audio)
    output_video_with_audio.write_videofile(args.output_video_path, codec='libx264', audio_codec='aac')
    
    print('time: {}'.format(time.time() - r_start_time))