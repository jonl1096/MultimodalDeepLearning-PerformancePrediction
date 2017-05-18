import csv
import sys
csv.field_size_limit(sys.maxsize)

inputfile = "shared_rep_at_LDA_withlabels_deeper.csv"
outputfile = "bimodal_AT_deep_with.csv"
rows_to_transform = set([1])
with open(inputfile, 'r') as ifile, open(outputfile, 'w') as ofile: 
	reader = csv.reader(ifile)
	writer = csv.writer(ofile)
	for row in reader:
		newrow = []
		for i,col in enumerate(row):
			if i in rows_to_transform:
				for number in list(eval(col)):
					newrow.append(number)
			else:
				newrow.append(col)
		writer.writerow(newrow)