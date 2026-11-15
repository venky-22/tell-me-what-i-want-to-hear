"""Figure 1: design schematic. python src/fig_design.py -> figs/fig0_design.png"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK='#1f2328'; MUT='#5b636e'; Y='#2a78d6'; N='#eb6834'
fig,ax=plt.subplots(figsize=(10,3.6),dpi=200); ax.set_xlim(0,100); ax.set_ylim(0,36); ax.axis('off')
def box(x,y,w,h): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.2',fc='white',ec='#c7ccd1',lw=1))
def arrow(x1,x2,y): ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle='-|>',color=MUT,lw=1.4))
box(1,6,24,25); ax.text(13,28.5,'1. User message',ha='center',fontsize=10,weight='bold',color=INK)
ax.text(13,23.5,'"Will X happen by\nDecember 31?"',ha='center',va='center',fontsize=8.5,color=INK,style='italic')
for i,(t,c) in enumerate([('neutral: (no framing)',MUT),('hope: "I really hope it does"',Y),('dread: "I hope it doesn\'t"',N),('stakes: "my savings are on it"',INK),('friend: "my friend hopes…"',MUT)]):
    ax.text(3,18-i*2.4,t,fontsize=7.2,color=c,va='center')
arrow(26,31,19)
box(32,6,26,25); ax.text(45,28.5,'2. Fixed evidence pool',ha='center',fontsize=10,weight='bold',color=INK)
for i,(L,s) in enumerate(zip('ABCDEFGH',['Y','N','N','Y','N','Y','Y','N'])):
    x=34+(i%4)*5.8; y=20.5-(i//4)*6.6; c=Y if s=='Y' else N
    ax.add_patch(FancyBboxPatch((x,y),4.6,4.8,boxstyle='round,pad=0.1,rounding_size=0.6',fc='white',ec=c,lw=1.6))
    ax.text(x+2.3,y+3.1,L,ha='center',fontsize=9,weight='bold',color=INK); ax.text(x+2.3,y+1.1,'YES' if s=='Y' else 'NO',ha='center',fontsize=6.5,color=c)
ax.text(45,9.6,'4 YES + 4 NO, matched in length\nand strength, fixed order',ha='center',va='center',fontsize=7,color=MUT)
arrow(59,64,19)
box(65,6,15,25); ax.text(72.5,28.5,'3. Model',ha='center',fontsize=10,weight='bold',color=INK)
ax.text(72.5,22,'SELECTED: B, E, G',ha='center',fontsize=8,family='monospace',color=INK)
ax.text(72.5,17.5,'PROBABILITY: 0.35',ha='center',fontsize=8,family='monospace',color=INK)
ax.text(72.5,10.5,'picks 3 of 8,\nthen a number',ha='center',va='center',fontsize=7,color=MUT)
arrow(81,85,19)
box(86,6,13,25); ax.text(92.5,28.5,'4. Measure',ha='center',fontsize=10,weight='bold',color=INK)
ax.text(92.5,22.5,'K',ha='center',fontsize=12,weight='bold',color=INK); ax.text(92.5,19.6,'YES picks:\nhope − dread',ha='center',fontsize=6.8,color=MUT)
ax.text(92.5,13.5,'D',ha='center',fontsize=12,weight='bold',color=INK); ax.text(92.5,10.6,'P(YES):\nhope − dread',ha='center',fontsize=6.8,color=MUT)
ax.text(50,1.8,'Evidence swap: re-show the model only its own 3 picks, under neutral or swapped framing, to separate what it reads (E) from how it weighs it (V).',ha='center',fontsize=7.5,color=MUT)
plt.savefig('figs/fig0_design.png',bbox_inches='tight',facecolor='white')
