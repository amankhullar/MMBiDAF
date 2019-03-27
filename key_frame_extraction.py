import os
import cv2
import subprocess

base_path = '/home/anish17281/NLP_Dataset/dataset/'

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def save_i_keyframes(video_fn, output_name):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        count_frame = 1
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = output_name + basename + '_i_frame_' + str(count_frame) + '.jpg'
            count_frame += 1
            cv2.imwrite(outname, frame)
            print ('Saved: '+outname)
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

def get_frames(filename):
    path = filename + "transcripts/"
    num_files = len([item for item in os.listdir(path) if os.path.isfile(os.path.join(path, item)) and '.txt' in item])
    for idx in range(1, num_files):
        output_name = filename + 'video_key_frames/' + str(idx) + '/'
        os.system('mkdir ' + output_name)
        save_i_keyframes(filename + 'videos/' + str(idx) + '.mp4', output_name)
        
if __name__ == '__main__':
    num_courses = 25
    for idx in range(1, num_courses):
        filename = base_path + str(idx) + '/'
        os.system('mkdir ' + filename + 'video_key_frames/')
        get_frames(filename)
