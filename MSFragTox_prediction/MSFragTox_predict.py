import sys, getopt

import zipfile
import os
import pandas as pd
import xgboost as xgb
import warnings
 
warnings.filterwarnings('ignore')

script_path = os.path.abspath(__file__)
root_path=os.path.dirname(script_path)


def get_index_list(fname,path):
        a=fname.split('.fpt')[0][-1]
        if a=='+':
            indexdir=os.path.join(path,'csi_fingerid.tsv')
            indexlist=pd.read_table(indexdir,header=0,usecols=['absoluteIndex'])
        elif a=='-':
            indexdir=os.path.join(path,'csi_fingerid.tsv')
            indexlist=pd.read_table(indexdir,header=0,usecols=['absoluteIndex'])
        return indexlist

def get_fingerprints(sirius_result_path):
    '''
    format {1:{},2:{}}
    '''
    path0=os.path.join(root_path,r"files\cis_fingerid_merged.csv")
    df_all_index=pd.read_csv(path0,header=0,index_col=None)

    fnames=os.listdir(sirius_result_path)
    flist=[s for s in fnames if '.' not in s]

    fps={}
    for idx,i in enumerate(flist):
        path1=os.path.join(sirius_result_path,i)
        fingerprintpath=os.path.join(path1,'fingerprints')
        try:
            fpname=os.listdir(fingerprintpath)
            fppath=fingerprintpath
        except:
            fingerprintpath1=os.path.join(path1,'fingerprintsz')
            if not os.path.exists(fingerprintpath1):
                os.makedirs(fingerprintpath1)
            zip_file=zipfile.ZipFile(fingerprintpath)
            zip_extract=zip_file.extractall(fingerprintpath1)
            zip_file.close()
            fpname=os.listdir(fingerprintpath1)
            fppath=fingerprintpath1
        fps1={}
        for idx2,fp in enumerate(fpname):
            df_index=get_index_list(fp,sirius_result_path)
            df_value=pd.read_table(os.path.join(fppath,fp),header=None,names=['platt_possibility'])
            df=pd.concat([df_index,df_value],axis=1)
            df_new=pd.merge(df_all_index,df,how='left',on='absoluteIndex') #fingerprints
            df_new.to_csv(os.path.join(sirius_result_path,'fingerprints.csv'),index=None)
            fps1[idx2]=df_new
        fps[idx]=fps1
    return(fps)

def predict(fp):
    assay_list = ['0_Aromatase','1_AhR','2_AR','3_ER','4_GR','5_TSHR','6_TR']
    ypred7=[]
    preoutcome7=[]
    for assay in assay_list:
        XGB_Model = xgb.Booster(model_file=os.path.join(root_path, 'files','models_for_7_assays', assay+'_stansmi_activity_spectrumid_vector.model'))
        df_test=pd.DataFrame(fp['platt_possibility']).T
        x=xgb.DMatrix(df_test)
        ypred = XGB_Model.predict(x)
        if ypred >= 0.5:
            outcome='active'
        else:
            outcome='inactive'
        ypred7.append(ypred[0])
        preoutcome7.append(outcome)
    return(ypred7,preoutcome7)

def usage():
	"""
    MSFragTox v.1.0 (predict the toxicity from SIRIUS result of MS/MS)

    usage:
    MSFragtox_prediction.py [-i|-input,=[input path]][-o|-output,=[output path]]
        [-h|--help][-v|--version]
    
    description:
    -i,-input  the output folder from SIRIUS predicting fingerprints from MS/MS file
    -o,-output  the path to output MSFragTox prediction result
    -h,--help 
    -v, --version 

    example:
    MSFragtox_prediction.py -i “D:/test1 6ppdq/6ppdq SIRIUS result” -o “D:/test1 6ppdq” 
    MSFragtox_prediction.py -h
    """


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"i:o:h:v",["inputfile=","outputfile=","help","version"])
        if opts == []:
            print(usage.__doc__)
    except getopt.GetoptError as err:
        print(err)
        print(usage.__doc__)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h',"--help"):
            print(usage.__doc__)
            sys.exit()
        elif opt in ('-v',"--version"):
            print('MSFragTox v.1.0')
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            inputfile = arg
        elif opt in ("-o", "--outputfile"):
            outputfile = arg
    if inputfile == '': 
        print('warning: input path is missing, please specify the input folder path!')
    else:
        print('1.successfully read file...')
        print(' the path of the SIRIUS predicted result is: ', inputfile)
        print('2.reading model files and predicting...')
        if outputfile == '':
            # inputfile=os.path.join(root_path,r"test1 6ppdq\6ppdq SIRIUS result")
            fps=get_fingerprints(inputfile)
            g=1
            for i in fps.keys():
                for j in fps[i].keys():
                    print('prediction result',g)
                    g+=1
                    porbvalue,outcome=predict(fps[i][j])
                    print('Toxicity endpoints: Aromatase\tAhR\tAR\tER\tGR\tTSHR\tTR')
                    print('Predicted toxicity possibility:',porbvalue)
                    print('Prediction result:',outcome)

        elif outputfile != '':
            # inputfile=os.path.join(root_path,r"D\MS\test1 6ppdq\6ppdq SIRIUS result")
            fps=get_fingerprints(inputfile)
            g=1  
            with open(os.path.join(outputfile,"prediction_result.txt"),"w") as f:
                for i in fps.keys():
                    for j in fps[i].keys():
                        f.write('Prediction result'+str(g)+'\n')
                        g+=1
                        probvalue,outcome=predict(fps[i][j])
                        probvalue1=map(str,probvalue)
                        outcome1=map(str,outcome)
                        f.write('Toxicity endpoints: Aromatase,AhR,AR,ER,GR,TSHR,TR\n')
                        f.write('Predicted toxicity possibility: '+','.join(probvalue1)+'\n'+'Prediction result: '+','.join(outcome1)+'\n')
            print('The output file is at: ',os.path.join(outputfile,'prediction_result.txt'))
        print('3.Prediction done!')

        # predict(inputfile,outputfile)

if __name__ == '__main__':
    main(sys.argv[1:])
