import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sys, matplotlib.pylab as plt
import time
from scipy.optimize import newton_krylov, anderson, broyden1, broyden2, excitingmixing, linearmixing, diagbroyden
import cProfile
import pstats

import plotly.graph_objects as go
np.random.seed(123)

# Laika solis
dt = 0.003
Pi = 3.14159265
# Magnētiskā lauka pilns precesijas periods ir Tau:
Tau = 1
# omega_h definīcija, izmantojot lauka precesijas periodu.
w_h = 2 * Pi / Tau
endlaiks = 5
laiksoli = int(endlaiks / dt)
laikskala = np.linspace(0, endlaiks, laiksoli)
angle_h = 0
# visas 'range' vērtības ir tas, cik smalku režģi veidot virsmai.
omegarange = 10
Hrange = 10
alfarange = 6
omega_arr = np.linspace(0.1, 2, omegarange)

# Masīvi, kuros glabājas 'range' skaits vienību, vienmērīgi izlīdzinātas nepieciešamajos vērtību intervālos.
att_arr = 1 / (omega_arr)
HHa_arr = np.linspace(0.1, 2, Hrange)
Harr = 1 / HHa_arr
alfa_harr = np.linspace(0.1 * Pi / 180, 90 * Pi / 180, alfarange)

# Magnētiskā lauka vērtības
w = np.array([0, 0, 1])
total = omegarange * Hrange * alfarange

# Mēs pētam vai bezdimensionālo proporciju vērtības dod stabilus vai nestabilus rezultātus
# simulācija pie vienas un tās pašas proporcijas izmēģina perturbētu un neperturbētu simulāciju. 
# ja tās nozīmīgi atšķirās pēc saprātīga iterāciju skaita, tad punkts fāzu telpā ir nestabils. 
# ja diference simulācijas sākumā ir aptuveni tāda pati 
# vai vismaz nav augusi eksponenciāli - metastabils
# Ja diference sākumā un beigās ir nodilusi uz ļoti mazu vērtību 
# t.i. trajektorijas ir saplūdušas - stabils. 

# def hlauks(alfa_h, moment):
#     hx = np.cos(angle_h + w_h * moment) * np.sin(alfa_h)
#     hy = np.sin(angle_h + w_h * moment) * np.sin(alfa_h)
#     hz = np.cos(alfa_h)
#     h = np.array([hx, hy, hz])
#     return h/np.linalg.norm(h)
@jit(nopython=True)
def hlauks(alfa_h, moment):
    hx = np.cos(angle_h + w_h * moment) * np.sin(alfa_h)
    hy = np.sin(angle_h + w_h * moment) * np.sin(alfa_h)
    hz = np.cos(alfa_h)
    h = np.array([hx, hy, hz])
    norm_h = np.sqrt(np.sum(h ** 2))
    return h / norm_h

# šo funkciju vairs nevajag, izmantoju iebūvēto moduļa aprēķinu 
# def modulis(vect):
#     return np.sqrt(vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2)

# funkcija valīdu sākuma nosacījumu atrašanai. 
# risināta ar visām pieejamajām optimizācijas metodēm ārpus cikla. 
# def Funk(n):
#     h = hlauks(alfa_h, 0)
#     p = np.cross(w, n)

#     return p - (k(n, h) / w_h)
@jit(nopython=True)
def Funk(n):
    h = hlauks(alfa_h, 0)
    p = np.cross(w, n)
    return p - (k(n, h) / w_h)

# Šī visa koda galvenais uzdevums ir iztestēt vienu parametru komplektu, bet ir arī otrs skripts, 
# kurš ar bisekcijas metodi meklē fāzu telpā metastabilo punktu virsmu, 
# ar pieņēmumu - starp stabilu un nestabilu punktu telpā [h\h0, w\w0] 
# vienmēr būs metastabils punkts. 

# galvenās funkcijas k, Fi, Teta tiek izsauktas tūkštošiem reižu tāpēc
# jebkurš optimizācijas ieguvums būs ar lielu pienesumu. 

# Šīs trīs funkcijas pašrakstītajā RK45 implementācijā ir primārais optimizācijas mērķis. 
# Sekundārais optimizācijas mērķis - mēģināt izmantot RK45 solveri no scipy, kas šī koda sākotnējā 
# rakstīšanas laikā 2017. gadā strādāja savādāk un nederēja šim use-case'am. 

# RK45 k_n koeficientu aprēķināšanas funkcija
# def k(n, h):
#     modarr = np.linalg.norm(n)
#     modhArr = np.linalg.norm(h)
#     FiCos = np.dot(n / modarr, h / modhArr) # iepriekšējā versijā šeit bija np.cos(np.arccos(...)) 
#     tetaCos = np.cos(Teta(FiCos))
#     w_a = w_h / att
#     return (w_a * tetaCos ** 2 / (FiCos + H * tetaCos)) * (h - n * FiCos)
@jit(nopython=True)
def k(n, h):
    modarr = np.sqrt(np.sum(n ** 2))
    modhArr = np.sqrt(np.sum(h ** 2))

    FiCos = np.dot(n / modarr, h / modhArr)
    tetaCos = np.cos(Teta(FiCos))
    w_a = w_h / att
    return (w_a * tetaCos ** 2 / (FiCos + H * tetaCos)) * (h - n * FiCos)

# Fi funkcija tiek izkļauta no izpildes, tās funkcionalitāti ieliku iekš funkcijas h. 
# atklāju arī, ka tiek izmantots np.cos(np.arccos(...)) 
# tāpēc vareja izkļaut arī 4 (četri RK45 koef) x 2 (cos, arccos) papildu funkciju callus katrā iterācijā.
# izvairamies arī papildu funkcijas Fi callošanas, kas nedaudz samazina overheadu. 

# def Fi(arr, hArr):
#     # moduļu aprēķināšanu var veikt ar iebūvēto funkciju
#     modarr = np.linalg.norm(arr)
#     modhArr = np.linalg.norm(hArr)
#     return np.arccos(np.dot(arr / modarr, hArr / modhArr)) # noņēmu apaļošanu, dārga operācija

@jit(nopython=True)
def Teta(fi):
    # 4. kārtas vienādojuma koeficienti izteikti ar zināmiem lielumiem.
    J = -H
    K = np.cos(fi)
    # L netiek izmantots, to var atmest. 
    p = np.array([-J ** 2, -2 * J * K, J ** 2 - 1, 2 * J * K, K ** 2])
    # Sakņu atrašana
    saknes = np.roots(p)
    rsaknes = saknes[np.isreal(saknes)].astype(float) # noņēmu apaļošanu, dārga operācija
    
    # Inversā substitūcija
    lenki = np.arccos(rsaknes)
    # Ievietošana enerģijas vienādojumā, lai salīdzinātu
    energijas = -0.5 * H * (np.cos(lenki)) ** 2 - np.cos(fi - lenki)
    # Interesē leņķis pie zemākās enerģijas -> tā būs grad E(phi), kurai sekos kustības vienādojums. 
    return lenki[np.argmin(energijas)]


# sākotnējā salīdzināšanas funkcijas 
# implementācija izdarīja trīs salīdzinājumus un tikai tad atgrieza rezultātu.
# returnu var izdarīt uzreiz, nedefinējot lokālo mainīgo k
# arī, tā kā mēs sagaidam ka stabilu un nestabilu punktu būs daudz vairāk kā metastabilu,
# var neizdarīt metastabilitātes salīdzinājumus ar a < x < b
def probe(x, pertmod):
    if (x > 1.05 * pertmod):
        print("UNSTABLE")
        return 1
    elif 0.95 * pertmod <= x:
        print("STABLE")
        return -1
    else:
        print("METASTABLE")
        return 0

ninit = np.array([0, 0, 1])

def AdVictoriam(H, att, alfa_h):
    laiks = 0
    # priekšlaicīgi tiek izveidoti rezultātu masīvi, kas ir efektīvi
    nd = np.zeros([laiksoli, 3, ])
    ndprim = np.zeros([laiksoli, 3, ])
    print("     ")
    Failed = False
    
    
    # Problēmas šeit ir, ka simulācijai vajag sākuma nosacījumus, 
    # kas jāatrod atrisinot VĒL VIENU diferenciālvienādojumu.
    # un tur risinājumi var neeksistēt. 
    # šī daļa, t.i. saķēdētie mēģinājumi lietot dažādos solverus sākuma nosacījumiem
    # patiesībā ir vislaikietilpīgākā daļa.
    for find in range(0, 3):
        # laika taupīšanas labad, izmēģināju tikai 3 dažādus komplektus. 
        # Lielākā daļa fāzu telpas režģa punktu ar vienu no metodēm atgriež rezultātus. 
        # Ir, protams, arī punkti bezdim. parametru telpā, kuriem sākuma nosacījumus vnk nevar atrast. 
        # Analītisku risinājumu tam, kuram punktam sākuma nosacījumu nostrādās un kuram ne nav vērts meklēt
        #, ja var vnk izlaist simulāciju vairākas reizes pārbaudot dažus punktus. 
        ninit = hlauks(alfa_h, find * Tau * 0.9 / 3)
        try:
            nvec = newton_krylov(Funk, ninit)
            print("Attempting Newton - Krylov Method")
        except:
            try:
                print("Newton - Krylov method did not converge, attempting to find initial conditions using Anderson mixing method,")
                nvec = anderson(Funk, ninit)
            except:
                try:
                    print("Attempting Broyden`s first Jacobian approximation,")
                    nvec = broyden1(Funk, ninit)
                except:
                    try:
                        print("Attempting Broyden`s second Jacobian approximation,")
                        nvec = broyden2(Funk, ninit)
                    except:
                        try:
                            print("Attempting to use tuned diagonal Jacobian approximation,")
                            nvec = excitingmixing(Funk, ninit)
                        except:
                            try:
                                print("Attempting to use a scalar Jacobian approxmation,")
                                nvec = linearmixing(Funk, ninit)
                            except:
                                try:
                                    print("Attempting Broyden Jacobian approximation")
                                    nvec = diagbroyden(Funk, ninit)
                                except:
                                    Failed = True
        if (Failed == False):
            break
    if (Failed == True):
        print("ALL ATTEMPTED METHODS FAILED, POINT FLAGGED AS A NON - CONVERGENT FOR INITIAL CONDITIONS")
        return np.array([H, att, alfa_h, 2])
    print("NVEC:")
    print(nvec)
    print(np.sqrt(nvec[0] ** 2 + nvec[1] ** 2 + nvec[2] ** 2))

    nvec = nvec / np.linalg.norm(nvec)
    pert = np.cross(-1 * w, hlauks(alfa_h, 0)) / 100
    ndnew = nvec
    nd[0] = ndnew / np.linalg.norm(ndnew)
    ndnewprim = nvec + (pert / np.linalg.norm(pert))
    ndprim[0] = ndnewprim / np.linalg.norm(ndnewprim)

    # Galvenais cikls (NEPERTURBĒTS) :
    nulltime = time.time()
    for i in range(0, laiksoli - 1):
        k1 = k(nd[i], hlauks(alfa_h, laiks))
        k2 = k(nd[i] + k1 / 2, hlauks(alfa_h, laiks + dt / 2))
        k3 = k(nd[i] + k2 / 2, hlauks(alfa_h, laiks + dt / 2))
        k4 = k(nd[i] + k3, hlauks(alfa_h, laiks + dt))

        nd[i + 1] = nd[i] + dt * (k1 + 2 * (k2 + k3) + k4) / 6
        nd[i + 1] = nd[i + 1] / np.linalg.norm(nd[i + 1])
        laiks = laiks + dt
    #  Sekundārais cikls (PERTRUBĒTS) :
    laiks = 0
    for i in range(0, laiksoli - 1):
        k1 = k(ndprim[i], hlauks(alfa_h, laiks))
        k2 = k(ndprim[i] + k1 / 2, hlauks(alfa_h, laiks + dt / 2))
        k3 = k(ndprim[i] + k2 / 2, hlauks(alfa_h, laiks + dt / 2))
        k4 = k(ndprim[i] + k3, hlauks(alfa_h, laiks + dt))

        ndprim[i + 1] = ndprim[i] + dt * (k1 + 2 * (k2 + k3) + k4) / 6
        ndprim[i + 1] = ndprim[i + 1] / np.linalg.norm(ndprim[i + 1])
        laiks = laiks + dt
    runtime = time.time()

    # Starpība starp abiem vektoriem - t.i. vai punkts ir stabils. 
    dev = ndprim - nd
    mod_dev = np.sqrt(dev[:, 0] ** 2 + dev[:, 1] ** 2 + dev[:, 2] ** 2)
    probe(mod_dev[laiksoli - 1], np.linalg.norm(pert))

    # Informatīvi, lai skatītos vai viss nav salūzis:
    print("nd[i]: ",nd[i])
    print("mod nd[i]: ",np.sqrt(nd[i][0] ** 2 + nd[i][1] ** 2 + nd[i][2] ** 2))
    print("ndprim: ",ndprim[i])
    print("mod ndprim[i]: ",np.sqrt(ndprim[i][0] ** 2 + ndprim[i][1] ** 2 + ndprim[i][2] ** 2))
    print("pavaditais laiks: ",runtime - nulltime)

    return nd, ndprim

# Šeit jāizvēlās trīs parametru vērtības
H, att, alfa_h = 2., 0.6, 0.38
strh = str(H)
stra = str(att)
strah = str(alfa_h)

# profileris tiek iedarbināts tikai uz pašas funkcijas izpildi.  
profiler = cProfile.Profile()
profiler.enable()
rezultati = AdVictoriam(H, att, alfa_h)
a = rezultati[0]
profiler.disable()
profiler.dump_stats("more_optimized.prof")
b = rezultati[1]

n = len(a)
# ar chatgpt salabots vizualizācijas kods: 
# Create traces for points in a and b
trace_a = go.Scatter3d(
    x=a[:, 0],
    y=a[:, 1],
    z=a[:, 2],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Neperturbēta sākumnosacījuma kustība'
)

trace_b = go.Scatter3d(
    x=b[:, 0],
    y=b[:, 1],
    z=b[:, 2],
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Perturbēta sākumnosacījuma kustība'
)
line_traces = []
for i in range(n):
    line_traces.append(
        go.Scatter3d(
            x=[a[i, 0], b[i, 0]],
            y=[a[i, 1], b[i, 1]],
            z=[a[i, 2], b[i, 2]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        )
    )

fig = go.Figure(data=[trace_a, trace_b] + line_traces)
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    ),
    title='3D Line Graph Connecting Arrays a and b',
    showlegend=True
)

fig.write_html("3dplot_more_optimiz.html")