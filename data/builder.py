from data.semg_datasets import TimeFeatureDataset, RowDataset

from data.base import BaseDataset

class DatasetBuilder:

    def __init__(self, args) -> None:
        self.args = args
    
    def builder(self, ) -> BaseDataset:
        dataset = None
        if self.args.features == 'row':
            dataset = RowDataset(self.args.data_path, 
                            self.args.subjects, 
                            self.args.classes,
                            self.args.window, 
                            self.args.stride, 
                            self.args.normalization,
                            self.args.denoise,
                            self.args.save_action_detect_result,
                            self.args.save_processing_result,
                            self.args.verbose)
        else:
            dataset = TimeFeatureDataset(
                            self.args.data_path,
                            self.args.subjects,
                            self.args.classes,
                            self.args.features,
                            self.args.window, 
                            self.args.stride, 
                            self.args.normalization,
                            self.args.denoise,
                            self.args.save_action_detect_result,
                            self.args.save_processing_result,
                            self.args.verbose)
        return dataset