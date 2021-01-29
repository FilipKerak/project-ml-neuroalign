from Bio import SeqIO
import numpy as np
medzera=-1
zhoda=1
nezhoda=-1


z01_01={}
z01_09={}
z09_01={}
z09_09={}
with open("sliced_01_ref01.paf",'r')as s:
    for r in s:
        h=r.split()
        if h[4]=='-':
            continue
        score=0
        for i in h:
            if 's1' in i:
                score=int(i.split(':')[2])
        z01_01[(h[0])]={'seq':h[5],'o':[int(h[2]),int(h[3])],'t':[int(h[7]),int(h[8])],'score':score}

with open("sliced_01_ref09.paf",'r')as s:
    for r in s:
        h=r.split()
        if h[4]=='-':
            continue
        for i in h:
            if 's1' in i:
                score=int(i.split(':')[2])
        z01_09[(h[0])]={'seq':h[5],'o':[int(h[2]),int(h[3])],'t':[int(h[7]),int(h[8])],'score':score}

with open("sliced_09_ref01.paf",'r')as s:
    for r in s:
        h=r.split()
        if h[4]=='-':
            continue
        for i in h:
            if 's1' in i:
                score=int(i.split(':')[2])
        z09_01[(h[0])]={'seq':h[5],'o':[int(h[2]),int(h[3])],'t':[int(h[7]),int(h[8])],'score':score}

with open("sliced_09_ref09.paf",'r')as s:
    for r in s:
        h=r.split()
        if h[4]=='-':
            continue
        for i in h:
            if 's1' in i:
                score=int(i.split(':')[2])
        z09_09[(h[0])]={'seq':h[5],'o':[int(h[2]),int(h[3])],'t':[int(h[7]),int(h[8])],'score':score}

ref01={}
for s in SeqIO.parse("reference01.fasta","fasta"):
   ref01[s.id]=str(s.seq)

ref09={}
for s in SeqIO.parse("reference09.fasta","fasta"):
   ref09[s.id]=str(s.seq)


res=[]
for s in SeqIO.parse("sliced_09.fastq","fastq"):
    if s.id not in z09_01 or s.id not in z09_09:
        continue
    n=len(s)
    rd=str(s.seq)
    
    start=z09_09[s.id]['t'][0]
    tar=ref09[z09_09[s.id]['seq']][start:start+n]
    sc1=z09_09[s.id]['score']
    
    start=z09_01[s.id]['t'][0]
    ftar=ref01[z09_01[s.id]['seq']][start:start+n]
    sc2=z09_01[s.id]['score']
   
    res.append(rd+" "+tar+" "+ftar+" "+str(sc1)+" "+str(sc2)+"\n")

for s in SeqIO.parse("sliced_01.fastq","fastq"):
    if s.id not in z01_01 or s.id not in z01_09:
        continue
    n=len(s)
    rd=str(s.seq)
    
    start=z01_01[s.id]['t'][0]
    tar=ref01[z01_01[s.id]['seq']][start:start+n]
    sc1=z01_01[s.id]['score']
    
    start=z01_09[s.id]['t'][0]
    ftar=ref09[z01_09[s.id]['seq']][start:start+n]
    sc2=z01_09[s.id]['score']
   

    res.append(rd+" "+tar+" "+ftar+" "+str(sc1)+" "+str(sc2)+"\n")

with open("data_variable",'w')as out:
    for r in res:
        out.write(r)
    
