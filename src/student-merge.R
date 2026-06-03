d1=read.table("../data/student-mat.csv",sep=",",header=TRUE)
d2=read.table("../data/student-por.csv",sep=",",header=TRUE)

d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
write.csv(d3,"../data/data_global_R.csv",row.names=FALSE)
print(nrow(d3)) # 382 students
