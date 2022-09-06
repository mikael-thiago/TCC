import numpy as np


class PreProcessing():
    @staticmethod
    def stack_frame(previous_frames, new_frame):
        number_of_frames = np.array(previous_frames).shape[2]
        return np.append(new_frame, previous_frames[:, :, :(number_of_frames-1)], axis=2)

    @staticmethod
    def replicate_frame(frame, number_of_frames=4):
        frame_list = [frame for i in range(number_of_frames)]
        frame = np.stack(tuple(frame_list), axis=2)
        return np.reshape([frame], (84, 84, number_of_frames))
