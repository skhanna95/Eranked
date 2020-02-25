import os
import operator
import optparse

class MetricCalc(object):

    def __init__(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    def maintain(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count

def readFiles(file_name):
    data_dict = {}
    with open(file_name) as lines:
        for line in lines:
            line_split = line.split()
            data_dict[line_split[0]] = {}
            for child in line_split[1].split(','):
                csplit = child.split('-')
                data_dict[line_split[0]][csplit[0]] = csplit[1]
                
    return data_dict

def calcScore(ref,op):
    metric = MetricCalc()
    
    for k,v in op.items():
        matches = 0
        
        for sk,sv in v.items():
            try: #penalizing if product not found
                if sv == ref[k][sk]:#increment if only match is found
                    matches += 1
            except:
                continue
                
        metric.maintain(matches/len(v))
        
    return round(metric.average,5)


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--devfile", dest='devfile', default='output/dev.out', help="Dev File Output Path")
    optparser.add_option("-t", "--testfile", dest="testfile", default='output/test.out', help="Test File  Output Path")
    optparser.add_option("-x", "--devreffile", dest='devreffile', default='data/reference/dev.out', help="Dev Reference File Path")
    optparser.add_option("-z", "--tesreffile", dest="tesreffile", default='data/reference/test.out', help="Test Reference File Path")
    optparser.add_option("-b", "--baseline", dest="baseline", default=False, help="Test Only Base Line Score")
    optparser.add_option("-p", "--backdir", dest="backdir", default='', help="directory to go back")


    (opts, _) = optparser.parse_args()

    baseline = opts.baseline
    if not isinstance(baseline, bool):
        if "true" in baseline.lower():
            baseline = True
        else:
            baseline = False

    if not baseline:
        if os.path.exists(opts.backdir+opts.devfile):
            if os.path.exists(opts.backdir+opts.devreffile):
                ref_dict = readFiles(opts.backdir+opts.devfile)
                op_dict = readFiles(opts.backdir+opts.devreffile)
                
                print('dev score:', calcScore(ref_dict,op_dict))
                
            else:
                print('No Dev Output File')
        else:
            print('Dev Reference File Missing')
        
        
        
    if os.path.exists(opts.backdir+opts.testfile):
        if os.path.exists(opts.backdir+opts.tesreffile):
            ref_dict_test = readFiles(opts.backdir+opts.testfile)
            op_dict_test = readFiles(opts.backdir+opts.tesreffile)
            
            print('test score:', calcScore(ref_dict_test,op_dict_test))

