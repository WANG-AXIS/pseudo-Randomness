load e.txt
fileID=fopen('e.txt')
C = textscan(fileID,'%s');
a1=length(C{1}{1});
a2=length(C{1}{10});
a3=length(C{1}{33335});
%str2num(C{1}{33335}) %89714262

Row_E_train=33334;
ESource=zeros(1,2000000);
for i=1:a1
    a=C{1}{1};
    ESource(1,i)=str2num(a(i));
end
for i=2:Row_E_train
    for j=1:a2
      a=(C{1}{i});
      ESource((i-2)*a2+57+j)=str2num(a(j));
    end
end


ESource(ESource<5)=0;
ESource(ESource>=5)=1;
sum(ESource)


