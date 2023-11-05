import torch
import numpy as np

class Metrics:
    def __init__(self, name_classes, mask=None, log_colors=True, device='cuda'):
    
        self.name_classes = name_classes    
        self.num_classes = len(name_classes)
        
        self.device = device
        self.log_colors = True
        
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=device)
        
        self.color_dict = {'cyan'  : '\033[96m',
                   'green' : '\033[92m',
                   'yellow': '\033[93m',
                   'red'   : '\033[91m'}

    def __genterate_cm__(self, pred, gt):                                                       #       preds                    
        mask = (gt >= 0) & (gt < self.num_classes)                                              #     +------- 
        combinations = self.num_classes*gt[mask] + pred[mask] # 0 <= comb <= num_classes^2-1    #   l | . . . 
        cm_entries = torch.bincount(combinations, minlength=self.num_classes**2)                #   b | . . .
        return cm_entries[:self.num_classes**2].reshape(self.num_classes, self.num_classes)                         #   s | . . .
        
    def add_sample(self, pred, gt):
        assert pred.shape == gt.shape, "Prediction and Ground Truth must have the same shape"
        self.confusion_matrix += self.__genterate_cm__(pred, gt) # labels along rows, predictions along columns
        
    def PA(self):
        # Pixel Accuracy (Recall) = TP/(TP+FN)
        return torch.diagonal(self.confusion_matrix)/self.confusion_matrix.sum(dim=1)
        
    def PP(self):
        # Pixel Precision = TP/(TP+FP)
        return torch.diagonal(self.confusion_matrix)/self.confusion_matrix.sum(dim=0)
        
    def IoU(self):
        # Intersection over Union = TP/(TP+FP+FN)
        return torch.diagonal(self.confusion_matrix)/(self.confusion_matrix.sum(dim=1)+self.confusion_matrix.sum(dim=0)-torch.diagonal(self.confusion_matrix))

    def percent_mIoU(self):
        return 100*self.nanmean(self.IoU())   

    def percent_prec(self):
        return 100*self.nanmean(self.PP())
        
    def percent_acc(self):
        return 100*self.nanmean(self.PA())

    @staticmethod
    def nanmean(tensor):
        m = torch.isnan(tensor)
        return torch.mean(tensor[~m])
        
    @staticmethod
    def nanstd(tensor):
        m = torch.isnan(tensor)
        return torch.std(tensor[~m])
        
    def color_tuple(self, val, c):
        return (self.color_dict[c], val, '\033[0m')
    
    @staticmethod
    def get_color(val, mean, std):
        if val < mean-std:
            return 'red'
        if val < mean:
            return 'yellow'
        if val < mean+std:
            return 'green'
        return 'cyan'
    
    def __str__(self):
        out = "="*46+'\n'
        out += "  Class           \t PA %\t PP %\t IoU %\n"
        out += "-"*46+'\n'
        
        pa, pp, iou = 100*self.PA(), 100*self.PP(), 100*self.IoU()
        mpa, mpp, miou = self.nanmean(pa), self.nanmean(pp), self.nanmean(iou)
        spa, spp, siou = self.nanstd(pa), self.nanstd(pp), self.nanstd(iou)
        for i, n in enumerate(self.name_classes):
            #if i+1 in self.mask:
            npa, npp, niou = pa[i], pp[i], iou[i]
            if self.log_colors:
                cpa, cpp, ciou = self.get_color(npa, mpa, spa), self.get_color(npp, mpp, spp), self.get_color(niou, miou, siou)
                tpa, tpp, tiou = self.color_tuple(npa, cpa), self.color_tuple(npp, cpp), self.color_tuple(niou, ciou)
                out += "  "+n+" "*(16-len(n))+"\t %s%.1f%s\t %s%.1f%s\t %s%.1f%s\n"%(*tpa, *tpp, *tiou)
            else:
                out += "  "+n+" "*(16-len(n))+"\t %.1f\t %.1f\t %.1f\n"%(npa, npp, niou)
        out += "-"*46+'\n'
        out += "  Average         \t %.1f\t %.1f\t %.1f\n"%(mpa, mpp, miou)
        out += "  Std. Dev.       \t %.1f\t %.1f\t %.1f\n"%(spa, spp, siou)
        out += "="*46+'\n'
        return out