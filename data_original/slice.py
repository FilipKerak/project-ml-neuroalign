from Bio import SeqIO
import numpy as np
towrite=[]
for s in SeqIO.parse("reads-barcode01.fastq","fastq"):
    x=max(0,len(s)//2-250)
    if len(s[x:x+600])<600:
        continue
    towrite.append(s[x:x+600])
SeqIO.write(towrite, "sliced_01_fixed.fastq", "fastq")
