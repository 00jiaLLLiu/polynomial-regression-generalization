import matplotlib.pyplot as plt
import pandas as pd


sigma =[0.01,0.1,1]
results = pd.read_csv("results.csv",index_col=0) # without regularization
#results = pd.read_csv("results_reg.csv",index_col=0)


#d=[3,10,17] N=[20,100]

#  N VS Ein,Eout,E_bias   sigma=0.1 
d=[3,5,10]  

# Creating figure to plot
fig,axes = plt.subplots(nrows=1,ncols=3,figsize = (17,6))
fig.suptitle("Ein,Eout,Ebias influenced by sample size for different d\n\nVariance={}\n".format(sigma[1]),size = 'xx-large')

dataN_y0=results[(results['Variance'] == sigma[1] ) & (results['d'] == d[0])]
dataN_y1=results[(results['Variance'] == sigma[1] ) & (results['d'] == d[1])]
dataN_y2=results[(results['Variance'] == sigma[1] ) & (results['d'] == d[2])]

for i in range(len(d)):
    data= results[(results['Variance'] == sigma[1] ) & (results['d'] == d[i])]
    axes[i].plot(data['N'],data['Ein'],label ="Ein")
    axes[i].plot(data['N'],data['Eout'],label ="Eout")
    axes[i].plot(data['N'],data['E_bias'],label ="E_bias")
    axes[i].set_xlabel("sample size",size = 'x-large')
    axes[i].set_ylabel("Mean Square Error",size = 'x-large')
    axes[i].set_title("d = {}\n".format(d[i]),size = 'x-large')
    axes[i].set_ylim(min(dataN_y0[['Ein','Eout','E_bias']].min().min(),dataN_y1[['Ein','Eout','E_bias']].min().min(),dataN_y2[['Ein','Eout','E_bias']].min().min()) -.1 ,
                     max(dataN_y0[['Ein','Eout','E_bias']].max().max(),dataN_y1[['Ein','Eout','E_bias']].max().max(),dataN_y2[['Ein','Eout','E_bias']].max().max()) +.1)
    axes[i].legend()

fig.tight_layout()
plt.show()




# d VS Ein,Eout,E_bias   sigma=0.01/ 0.1 /1
N=[20,100]
fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (14,6))

fig.suptitle("Ein,Eout,Ebias influenced by model complexity\n\nVariance={}\n".format(sigma[1]),size = 'x-large')

dataD_y0=results[(results['N'] == N[0]) & (results['Variance'] == sigma[1]) & (results['d'] > 0)]
dataD_y1=results[(results['N'] == N[1]) & (results['Variance'] == sigma[1]) & (results['d'] > 0)]

for i in range(len(N)):
    data= results[ (results['N'] == N[i]) & (results['Variance'] == sigma[1]) & (results['d'] > 0)] 
    axes[i].plot(data['d'],data['Ein'],label ="Ein")
    axes[i].plot(data['d'],data['Eout'],label ="Eout")
    axes[i].plot(data['d'],data['E_bias'],label ="E_bias")
    axes[i].set_xlabel("model complexity",size = 'x-large')
    axes[i].set_ylabel("Mean Square Error",size = 'x-large')
    axes[i].set_title("sample size= {}\n".format(N[i]),size = 'x-large')
    axes[i].set_ylim(min(dataD_y0[['Ein','Eout','E_bias']].min().min(),dataD_y1[['Ein','Eout','E_bias']].min().min()) -.1 ,
                     max(dataD_y0[['Ein','Eout','E_bias']].max().max(),dataD_y1[['Ein','Eout','E_bias']].max().max()) + .1)
    axes[i].legend()

fig.tight_layout()
plt.show()


# sigma VS Ein,Eout,E_bias   N=100 d=8

fig,axes=plt.subplots(1,3,figsize = (15,6))
fig.suptitle("Ein,Eout,Ebias influenced by noise level\n sample size(N)= 100  model complexity(d)=8\n ",size = 'x-large')

dataS_y0=results[(results['Variance'] == sigma[0]) & (results['N'] == 100 )  & (results['d'] == 8) ]
dataS_y1=results[(results['Variance'] == sigma[1]) & (results['N'] == 100 )  & (results['d'] == 8) ]
dataS_y2=results[(results['Variance'] == sigma[2]) & (results['N'] == 100 )  & (results['d'] == 8) ]

for i in range(len(sigma)):  
    data= results[(results['Variance'] == sigma[i]) & (results['N'] == 100 )  & (results['d'] == 8) ]
    axes[i].bar(sigma[i],data['Ein'],width =.4)
    axes[i].bar(sigma[i]+1,data['Eout'],width =.4)
    axes[i].bar(sigma[i]+2,data['E_bias'],width =.4)
    
    axes[i].set_title('Sigma = {}\n'.format(sigma[i]),size = 'x-large')
    axes[i].set_ylim(0,max(dataS_y0[['Ein','Eout','E_bias']].max().max(),dataS_y1[['Ein','Eout','E_bias']].max().max(),dataS_y2[['Ein','Eout','E_bias']].max().max() ) +.1 )
    axes[i].set_xticks(axes[i].get_xticks())
    axes[i].set_xticklabels(['','Ein','','Eout','','E_bias',''],size = 'x-large')
fig.tight_layout()
plt.show()
