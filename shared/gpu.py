import tensorflow as tf


class GpuUtils():
    @staticmethod
    def limit_memory_usage(max_memory: int):
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            print('Limiting GPU memory usage to {} mbs'.format(max_memory))

            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=max_memory)])
            except RuntimeError as e:
                print('Some error ocurred trying to limit GPU memory usage')
                print(e)
