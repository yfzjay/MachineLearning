import trees
import treePlotter
def main():
    fr=open("../file/Ch03/lenses.txt")
    Map=[line.strip().split('\t') for line in fr]
    print(Map)
    Label=['age','prescript','astigmatic','tearRate']
    Tree=trees.createTree(Map,Label)
    print(Tree)
    treePlotter.createPlot(Tree)
if __name__=='__main__':
    main()
