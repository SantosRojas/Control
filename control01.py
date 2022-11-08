from scipy.signal import lti,bode
import numpy as np
import matplotlib.pyplot as plt
#==================Comentarios=========
#Autor: Santos Herminio Rojas Gutierrez
#Es libre para su uso y mejoras
#======================================

def tf(num,den,r=0,method='Pade'):
    '''
    num: Numerador de la funcion de transferencia del sistema
    den: Denominador de la funcion de transferencia del sistema
    r: Retardo del systema
    methodo: Metodo de aproximacion del retardo (default: Pade)
    '''
    if r>0:
        if method=='Pade':
            num_p=[(r)**2,-6*r,12]
            den_p=[(r)**2,+6*r,12]
            num=np.convolve(num,num_p)
            den=np.convolve(den,den_p)
        if method=='AR':
            num_p=[0.0783*(r)**2,-0.4986*r,1]
            den_p=[0.0783*(r)**2,+0.4986*r,1]
            num=np.convolve(num,num_p)
            den=np.convolve(den,den_p)
        
        if method=='ARop':
            num_p=[0.091*(r)**2,-0.496*r,1]
            den_p=[0.091*(r)**2,+0.496*r,1]
            num=np.convolve(num,num_p)
            den=np.convolve(den,den_p)
            
    return lti(num,den)
    
def serie(sys1,sys2):
    '''
    Para tf en serie
    sys1 y sys2 son control.tf()
    '''
    num=np.convolve(sys1.num,sys2.num)
    den=np.convolve(sys1.den,sys2.den)
    return lti(num,den)

def feedback(sys1,sys2):
    '''
    Para tf con retroalimentacion negativa
    sys1 y sys2 son control.tf o scipy.signal.lti
    sys1: Funcion de transferencia de camino directo
    sys2:Funcion de transferencia de realimentacion negativa
    '''
    num=np.convolve(sys1.num,sys2.den)
    den1=np.convolve(sys1.num,sys2.num)
    den2=np.convolve(sys1.den,sys2.den)
    n=len(den1)
    m=len(den2)
    
    if n>m:
        den2=np.append(den2[::-1],np.zeros(n-m))[::-1]
        
    if m>n:
        den1=np.append(den1[::-1],np.zeros(m-n))[::-1]
        
    den=den1+den2
    
    return lti(num,den)

def plot_step(*args, T=None, N=None,cs=3,params='vs',legend_params=False,only_params=False,text_params=False,anotation=False,**kwargs):
    '''
    *args: [sys,{X0:X0,
                    c:color,
                    lw:linewidth,
                    label:label}]

    sys es control.tf() o sicpy.signal.lti
                  
                    
    X0: Estado inicial del sistema.

    T:[ti,tf,n] Vector de tiempo (lista,ti:tiempo inicial,tf:tiempo final,n:numero de puntos).

    N: Numero de puntos de T (en caso no se de T).
    
    cs:cifras significativas.
    
    params: str, define que parametros se analizaran (solo se pueden plotear 'tp,ts,vp,vs') {
        'all': todos los parametros('tp,ts,vp,mp,tl,vs')
        'tp,vs': tiempo pico mas valor estable
        'tp,ts':tiempo pico mas tiempo de establecimiento
        ...
        'tp,ts,vp,':tiempo pico mas tiempo de establecimiento mas valor pico
    }

    legend_params: Bool, define si se añale los parametros a una legenda, default False.
    
    text_params: Bool, define si se muestra los parametros como texto, default False.

    only_params: Bool, define si solo se muesta los parametros en formato de texto. Default False
    
    **kwargs:title,xlabel,ylabel,loc
    >>loc:
    'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 
    'center right', 'lower center', 'upper center', 'center'
    
    '''

    #=====================Inicializacion de vaiables=======
    if only_params:
        text_params=True
        
    args_plot={'title':'Respuesta al escalon unitario','xlabel':'Tiempo(S)','ylabel':'Magnitud','loc':'best'}
    for key,_ in kwargs.items():
        if key in args_plot.keys():
            args_plot[key]=kwargs[key]
        else:
            raise ValueError(f'''La llave {key} es desconocida. Utilizar 'title','xlabel','ylabel','loc' ''')

    
    if not T is None:
        if T[1]<T[0]:
            raise ValueError('El valor de tf debe ser mayor a ti en T=[ti,tf,n]')
        else:
            T=np.linspace(T[0],T[1],T[2])
        
    prms=['tp','ts','vp','mp','tl','vs']
    c=['b','r','g','c','m','y']
    labels=[]
    SYS=[]#lista de funciones de transferencia
    ARG=[]#lista de argumentos para las graficas
    Tx=[]
    Yx=[]
    Tp=[]
    Ts=[]
    Vp=[]
    Vs=[]

    #===============Verificacion de los parametros ingresados=====
    if not params is None:
        list_params=params.split(',')
        
        if len(list_params)==1:
            if list_params[0]=='all':
                list_params=prms
            elif list_params[0] in prms:
                pass
            else:
                raise ValueError(f'El parametro {list_params[0]} es desconocido, utilizar tp,ts,vp,mp,vs,tl')
        else:
            for i in list_params:
                if not i in prms:
                    raise ValueError(f'El parametro {list_params[0]} es desconocido, utilizar tp,ts,vp,mp,vs,tl')
                    
    else:
        list_params=[]
    
        
    
    #====================Separamos tf de  parametros===================
    for s in args:
        try:
            if len(s)>1:
                SYS.append(s[0])
                ARG.append(s[1])
            else:
                SYS.append(s[0])
                ARG.append({'X0':None})
        except:
            SYS.append(s)
            ARG.append({'X0':None})
            
      
        
    #====================Generamos una lista de labels por default===================
    for n in range(len(SYS)):
        labels.append(f'Resp_tem_{n+1}')
        
    #====================Limpiamos los argumentos===================
    n=0
    for i in ARG:
        if not 'X0' in i:
            i['X0']=None

        if not 'c' in i:
            i['c']=c[n]
            
        if not 'label' in i:
            i['label']=labels[n]
            
        if not 'lw' in i:
            i['lw']=1.5
            
        n+=1
    #====================Calculos de los parametros===================
    
    for i,sys in enumerate(SYS):
        t,y=sys.step(X0=ARG[i]['X0'],T=T,N=N)
        y_r=list(y)
        t_r=list(t)
        t_r.reverse()
        y_r.reverse()
        if text_params:
            print(f"=========== PARAMETROS DE {ARG[i]['label']} ===============")
            
        if 'tp' in list_params:
            tp=round(t[list(y).index(y.max())],cs)#tiempo pico
            Tp.append(tp)
            if text_params:
                print(f'Tiempo pico(S) = {tp}')
            
        if 'ts' in list_params:
            vs=sys.num[-1]/sys.den[-1]
            g=round(abs(vs-y[0]),cs)
            ts=0 #tiempo de establecimiento
            for k,l in zip(y_r,t_r):
                try:
                    if abs(k-g)<=0.02*g:
                        ts=l
                    else:
                        ts=round(ts,cs)
                        break
                except Exception as e:
                    raise ValueError('El sistema es inestable')
            Ts.append(ts)
            if text_params:
                print(f'Tiempo de establecimiento(S) = {ts}')
            
        if 'tl' in list_params:
            vs=sys.num[-1]/sys.den[-1]
            try:
                for k,l in zip(y,t):
                    if k<=vs*0.1:
                        tli=l
                    else:
                        tli=round(tli,cs)
                        break
                for k,l in zip(y,t):
                    if k<=vs*0.9:
                        tlf=l
                    else:
                        tlf=round(tlf,cs)
                        break
                        
                tl=tlf-tli 
                tl=round(tl,cs)#Tiempo de levantamiento
                if text_params:
                    print(f'Tiempo de levantamiento = {tl}')
                
            except Exception as e:
                raise ValueError("El sistema no presenta un tiempo de levantamiento")
                
        if 'mp' in list_params:
            mp=abs(y.max()-vs)/vs
            mp=round(mp,cs)#Maximo sobreimpulso
            if text_params:
                print(f'Maximo sobreimpulso = {mp}')
        
        if 'vp' in list_params:
            vp=round(y.max(),cs)
            Vp.append(vp)
            if text_params:
                print(f'Valor pico = {vp}')
        
        if 'g' in list_params:
            g=round(abs(vs-y[0]),cs)#ganancia
            if text_params:
                print('Ganancia = ',g)
            
        if 'vs' in list_params:
            vs=sys.num[-1]/sys.den[-1]
            vs=round(vs,cs)#valor estable
            Vs.append(vs)
            if text_params:
                print('Valor estable = ',vs)
            
        
        if not only_params:
            Tx.append(t)
            Yx.append(y)
            
        
        
    if not only_params:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(SYS)):
            line, = ax.plot(Tx[i],Yx[i],ARG[i]['c'],label=ARG[i]['label'],linewidth=ARG[i]['lw'])

            if legend_params:
                legends=[]
                if 'vp' in list_params:
                    legends.append(f'$Vp{i+1}={Vp[i]}$')
                else:
                    legends.append(None)
                    
                if 'tp' in list_params:
                    legends.append(f'$tp{i+1}={Tp[i]}$')
                else:
                    legends.append(None)
                
                if 'ts' in list_params:
                    legends.append(f'$ts{i+1}={Ts[i]}$')
                else:
                    legends.append(None)
                    
            else:
                legends=[None]*3
            
            
            if 'vs' in list_params:
                plt.plot([Tx[i][0],Tx[i][-1]],[Vs[i]]*2,'grey',linestyle='dotted')
                if anotation:
                    ax.annotate(f'{Vs[i]}', xy=(0.2,Vs[i]))
            if 'vp' in list_params:
                plt.plot([Tx[i][0],Tp[i]],[Vp[i]]*2,'-.',label=legends[0])
                if anotation:
                    ax.annotate(f'{Vp[i]}', xy=(Tp[i],Vp[i]))
            if 'tp' in list_params:
                plt.plot([Tp[i]]*2,[Yx[i][0],Yx[i].max()],'-.',label=legends[1])
                if anotation:
                    ax.annotate(f'{Tp[i]}', xy=(Tp[i],0.2))
            if 'ts' in list_params:
                plt.plot([Ts[i]]*2,[Yx[i][0],Yx[i][-1]],'-.',label=legends[2])
                if anotation:
                    ax.annotate(f'{Ts[i]}', xy=(Ts[i],0.2))
        
        plt.plot([Tx[0][0],Tx[i][-1]],[1]*2,'k',label='Setpoint')
        plt.title(args_plot['title'])
        plt.xlabel(args_plot['xlabel'])
        plt.ylabel(args_plot['ylabel'])
        plt.legend(loc=args_plot['loc'])
        plt.show()

        
def PI(sys,m,ts):
    '''
    sys=tf
    m=sobre impulso en decimal.
    ts: tiempo de establecimiento.
    '''
    z=1/np.sqrt(1+(np.pi/np.log(m))**2)
    w=4/(z*ts)
    N=sys.num
    D=sys.den

    s1=complex(-w*z,w*np.sqrt(1-z**2))
    s2=complex(-w*z,-w*np.sqrt(1-z**2))

    D1=np.polyval(D,s1)
    D2=np.polyval(D,s2)
    N1=np.polyval(N,s1)
    N2=np.polyval(N,s2)

    kp=(s2*D2*N1-s1*D1*N2)/((s1-s2)*N1*N2)
    ki=-kp*s1-D1*s1/N1
    kp=kp.real
    ki=ki.real
    Gdir=serie(sys,tf([kp,ki],[1,0]))
    G=feedback(Gdir,tf(1,1))
    return G


def LGR(sys,xlim=None,ylim=None,fc='cyan',grid=False,anotation=True):
    real=[]
    imag=[]
    p=sys.poles
    z=sys.zeros
    k=np.linspace(0.01,300,3000)
    fig = plt.figure(facecolor=fc)
    ax = fig.add_subplot(111)
    for i in k:
        ser=serie(sys,tf([i],[1]))
        fed=feedback(ser,tf([1],[1]))
        ps=fed.poles
        real.append(ps.real)
        imag.append(ps.imag)
    ax.plot(real,imag,linewidth=2)
    ax.plot(p.real,p.imag,c='red',marker='x',ms=10,ls='',linewidth=5)# graficamos los polos en lazo directo
    ax.plot(z.real,z.imag,c='blue',marker='o',ms=10,ls='',linewidth=5)#graficamos los ceros en lazo directo
    if anotation:
        for i in p:
            ax.annotate(f'{np.round(i,3)}', xy=(i.real+0.02,i.imag+0.02))

        for j in z:
            ax.annotate(f'{np.round(j,3)}', xy=(j.real+0.02,j.imag+0.02))
    
    plt.title('LGR')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.grid(grid)
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)
    plt.show()

def Bode(sys,ax=None,):
    w,mag,fase=bode(sys)
    if ax is None:
        fig,ax=plt.subplots(2,1)

    ax[0].semilogx(w,mag)
    ax[1].semilogx(w,fase)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_xlabel('Frecuencia')
    ax[1].set_xlabel('Frecuencia')
    ax[0].set_ylabel('Magnitud (db)')
    ax[1].set_ylabel('Fase')
    plt.show()


