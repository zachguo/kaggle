import math, sys

def zygmoid(x): return 1 / (1 + math.exp(-x))

if __name__ == '__main__':
	tag = sys.argv[1]
	with open("output/cv_ensemble/submission_%s.csv" % tag, "wb") as outfile:
		outfile.write("Id,Predicted\n")
		for line in open("output/cv_ensemble/vw/predictions_%s.txt" % tag, "rb"):
			row = line.strip().split(" ")
			outfile.write("%s,%f\n"%(row[1], zygmoid(float(row[0]))))
