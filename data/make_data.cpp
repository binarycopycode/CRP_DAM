#include<bits/stdc++.h>
using namespace std;

mt19937 rnd(time(0));
const int maxl=1e5+10;

int a[maxl];

string str(int x)
{
	string ret="";
	while(x)
	{
		ret=(char)(x%10+'0')+ret;
		x/=10;
	}
	return ret;
}

int main()
{
	int b=40,s=6,t=5,n=18;
	puts("please input batch , s , t , n");
	scanf("%d%d%d%d",&b,&s,&t,&n);
	
	FILE *f;
	string fs="test_"+str(n)+"_"+str(s)+"_"+str(t)+".txt";
	const char *fss=fs.c_str(); 
	f=fopen(fss,"w");
	fprintf(f,"%d %d %d %d\n",b,s,t,n);
	
	FILE *fam;
	string fams="test_am_"+str(n)+"_"+str(s)+"_"+str(t)+".txt";
	const char *famss=fams.c_str(); 
	fam=fopen(famss,"w");
	fprintf(fam,"%d %d %d %d\n",b,s,t,n);
	
	while(b--)
	{
		for(int j=1;j<=n;j++)
			a[j]=j;
		shuffle(a+1,a+1+n,rnd);
		for(int j=1;j<=n;j+=t-2)
		{
			fprintf(f,"%d",t-2);
			fprintf(fam,"%d",t-2);
			for(int k=0;k<t-2;k++)
			{
				//in rcrp-idbb, smaller number first
				fprintf(f," %d",a[j+k]);
				//in rcrp-am  ,bigger number first
				fprintf(fam," %d",n-a[j+k]+1);
			}
			fprintf(f,"\n");
			fprintf(fam,"\n");
		}
		
	} 
	return 0;
}
