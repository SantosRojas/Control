from scipy.signal import lti
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
    sys1 y sys2 son scipy.signal.lti
    sys1: Funcion de transferencia de camino directo
    sys2:Funcion de transferencia de realimentacion negativa
    '''
    num=np.convolve(sys1.num,sys2.den)
    den=np.convolve(sys1.num,sys2.num)+np.convolve(sys1.den,sys2.den)
    return lti(num,den)

def plot_step(*args, T=None, N=None,cs=3,params='vs',plot_params=False,legend_params=False,only_params=False,text_params=False,**kwargs):
    '''
    sys es control.tf()
    
    
    cs:cifras significativas.
    
    *args: [sys,{X0:X0,
                    c:color,
                    lw:linewidth,
                    label:label}]
                    
                    
    X0: Estado inicial del sistema

    T:[ti,tf,n] Vector de tiempo (lista,ti:tiempo inicial,tf:tiempo final,n:numero de puntos).

    N: Numero de puntos de T (en caso no se de T).
    
    '''
    #=====================Inicializacion de vaiables=======
    args_plot={'title':'Respuesta al escalon unitario','xlabel':'Tiempo(S)','ylabel':'Magnitud','loc':'best'}
    for key,value in kwargs.items():
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
    keys=['X0','lw','label','c']
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
            print(f"=========== Parametros de {labels[i]} ===============")
            
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
        for i in range(len(SYS)):
            plt.plot(Tx[i],Yx[i],ARG[i]['c'],label=ARG[i]['label'],linewidth=ARG[i]['lw'])

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
            if 'vp' in list_params:
                plt.plot([Tx[i][0],Tp[i]],[Vp[i]]*2,'-.',label=legends[0])
            if 'tp' in list_params:
                plt.plot([Tp[i]]*2,[Yx[i][0],Yx[i].max()],'-.',label=legends[1])
            if 'ts' in list_params:
                plt.plot([Ts[i]]*2,[Yx[i][0],Yx[i][-1]],'-.',label=legends[2])
        
        plt.plot([Tx[0][0],Tx[i][-1]],[1]*2,'k',label='Setpoint')
        plt.title(args_plot['title'])
        plt.xlabel(args_plot['xlabel'])
        plt.ylabel(args_plot['ylabel'])
        plt.legend(loc=args_plot['loc'])
        plt.show()
