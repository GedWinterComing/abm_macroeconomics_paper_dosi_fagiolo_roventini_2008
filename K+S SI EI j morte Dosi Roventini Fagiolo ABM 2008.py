# -*- coding: utf-8 -*-
"""
Created on Dec 2023

@author: Gabriele Coppini
Python 3.5
"""
#==============================================================================
#                 Agent-Based Model (ABM)
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(4+4) # così con lo stesso seed (qua 420 ma può essere qualunque numero) ottieni sempre gli stessi numeri "casuali". Oppure puoi commentare questa riga, ma i risultati non saranno più riproducibili perchè ogni volta otterrai numeri casuali diversi! # Fixing random state for reproducibility
# ctrl + 4 (tasto numero) per commentare più righe col bordo speciale grafica particolare
# import sys # per fermare esecuzione programma tramite sys.exit()

import statsmodels.api as sm
# BAXTER-KING FILTER (1999)

def filter_baxter_king_1999(serie_originaria, p_periodo_T_piccolo, q_periodo_T_grande, K_buttare_via):
    # ESEMPIO: filter_baxter_king_1999(serie_originaria, 6, 32, 12):
    omega_fine = (2*np.pi)/p_periodo_T_piccolo
    omega_inizio = (2*np.pi)/q_periodo_T_grande
    lunghezza_serie_originaria = len(serie_originaria)
    if lunghezza_serie_originaria < (2*K_buttare_via + 2):
        return print("ERRORE! La serie è troppo corta per questo K scelto! Siccome andranno butatti via K dati all'inizio e alla fine, la serie deve contenere almeno K*2 + 1 dati")
    else:
        pesi_filtro_vettore_omega_fine = np.array([])
        peso_zero_omega_fine = omega_fine/np.pi
        pesi_filtro_vettore_omega_fine = np.append(pesi_filtro_vettore_omega_fine, peso_zero_omega_fine)
        pesi_filtro_vettore_omega_inizio = np.array([])
        peso_zero_omega_inizio = omega_inizio/np.pi
        pesi_filtro_vettore_omega_inizio = np.append(pesi_filtro_vettore_omega_inizio, peso_zero_omega_inizio)
        for k_piu_uno_zero_aggiunto in range(K_buttare_via):
            # il problema è che nelle formule indice_somma dovrebbe partire da 1 e non da zero come fa Python, anche perchè il termine zero l'abbiamo considerato sopra a parte, per cui bisogna aggiungere un 1
            k_piu_uno_zero_aggiunto = 1+k_piu_uno_zero_aggiunto
            # assumendo che i pesi siano simmetrici quindi, a parte il peso(0), tutti gli altri vale che peso(-K)=peso(+K)
            peso_k_piu_uno_omega_fine = np.sin(k_piu_uno_zero_aggiunto*omega_fine)/(k_piu_uno_zero_aggiunto*np.pi)
            pesi_filtro_vettore_omega_fine = np.append(pesi_filtro_vettore_omega_fine, peso_k_piu_uno_omega_fine)
            peso_k_piu_uno_omega_inizio = np.sin(k_piu_uno_zero_aggiunto*omega_inizio)/(k_piu_uno_zero_aggiunto*np.pi)
            pesi_filtro_vettore_omega_inizio = np.append(pesi_filtro_vettore_omega_inizio, peso_k_piu_uno_omega_inizio)
        theta_filtro_omega_fine = (1 - pesi_filtro_vettore_omega_fine[0] - 2*sum(pesi_filtro_vettore_omega_fine[1:]) )/(1+2*K_buttare_via) # in realtà questo è sbagliato poichè è 1-.. non dovrebbe esserci ma dovrebbe essere direttamente -.. però siccome poi fai la differenza dei due theta i 2 uno si cancellano a vicenda
        theta_filtro_omega_inizio = (1 - pesi_filtro_vettore_omega_inizio[0] - 2*sum(pesi_filtro_vettore_omega_inizio[1:]) )/(1+2*K_buttare_via)
        pesi_filtro_vettore_sono_K_piu_uno = (pesi_filtro_vettore_omega_fine - pesi_filtro_vettore_omega_inizio) + (theta_filtro_omega_fine - theta_filtro_omega_inizio)
        # i pesi hanno la proprietà che la sommatoria da -K a +K è nulla e sono simmetrici: pesi_filtro_vettore_sono_K_piu_uno[0] + 2*sum(pesi_filtro_vettore_sono_K_piu_uno[1:])
        serie_pulita_e_accorciata = np.array([])
        for contatore in range(lunghezza_serie_originaria - 2*K_buttare_via):
            termine_zero = pesi_filtro_vettore_sono_K_piu_uno[0] * serie_originaria[contatore + K_buttare_via]
            somma_termine_serie_pos_contatore = termine_zero
            for indice_somma in range(K_buttare_via):
                indice_somma_corretto = 1+indice_somma # siccome non deve partire da zero ma da 1 e arrivare a K compreso e non a (K-1)
                somma_termine_serie_pos_contatore = somma_termine_serie_pos_contatore + (serie_originaria[contatore + K_buttare_via - indice_somma_corretto] + serie_originaria[contatore + K_buttare_via + indice_somma_corretto]) * pesi_filtro_vettore_sono_K_piu_uno[indice_somma_corretto]
            serie_pulita_e_accorciata = np.append(serie_pulita_e_accorciata, somma_termine_serie_pos_contatore)
        return serie_pulita_e_accorciata


def lagatore_restit_2_serie_lagate_stessa_len(X,K):
    lunghezza_originale = len(X)
    lunghezza_accorciata = lunghezza_originale - K
    x = np.array([])
    y = np.array([])
    for posizionatore in range(lunghezza_accorciata):
        posizione_x = K + posizionatore
        x = np.append(x, X[posizione_x])
        y = np.append(y, X[posizionatore])
    return x,y

def avantatore_restit_2_serie_lead_stessa_len(X,K):
    lunghezza_originale = len(X)
    lunghezza_accorciata = lunghezza_originale - K
    x = np.array([])
    y = np.array([])
    for posizionatore in range(lunghezza_accorciata):
        # in pratica è come il lagatore, ma invertito. Infatti nel paper i risultati per il GDP sono simmetrici, ovvero t-k=t+k
        posizione_y = K + posizionatore
        x = np.append(x, X[posizionatore])
        y = np.append(y, X[posizione_y])
    return x,y

def lagatore_doppio_input_restit_2_serie_lagate_stessa_len(X,Y,K):
    lunghezza_originale = len(Y)
    lunghezza_accorciata = lunghezza_originale - K
    x = np.array([])
    y = np.array([])
    for posizionatore in range(lunghezza_accorciata):
        posizione_x = K + posizionatore
        x = np.append(x, X[posizione_x])
        y = np.append(y, Y[posizionatore])
    return x,y

def avantatore_doppio_input_restit_2_serie_lead_stessa_len(X,Y,K):
    lunghezza_originale = len(Y)
    lunghezza_accorciata = lunghezza_originale - K
    x = np.array([])
    y = np.array([])
    for posizionatore in range(lunghezza_accorciata):
        # in pratica è come il lagatore, ma invertito. Infatti nel paper i risultati per il GDP sono simmetrici, ovvero t-k=t+k
        posizione_y = K + posizionatore
        x = np.append(x, X[posizionatore])
        y = np.append(y, Y[posizione_y])
    return x,y
#%%
# PARAMETRI e CONDIZIONI INIZIALI
# condizioni iniziali
F1_numero_consumption_industry = 200
F2_numero_capital_industry = 50
K_j_0_capital_stock_iniziale_per_azienda_beni = 800
average_labor_productivity_iniziale = 1
average_firm_labor_productivity_iniziale_supposta_da_me = 1
NW_i_j_0_liquid_assets = 10000
mi_0_mark_up_iniziale = 0.3
r_interest_rate = 0
cpi_0_consumer_price_index_iniziale = 1.3 # ovvero p_j(t=0)
L_0_labor_supply_popolazione_lavoratori = 1000000
w_0_market_wage = 1
HC_i_0_clienti_iniziali = 7 #stai facendo con 7 le simulazioni chiusura progetto ma ci sono tantissimi restart capital e statistiche degli investimenti bassi, inoltre occupazione invece di annullarsi satura
#HC_i_0_clienti_iniziali = 20
# nel paper non c'è scritto quindi l'ho scelto io a mio parere, ma se è troppo basso molte consumption j potrebbero non ricevere nemmeno una brochure e perciò non avere _i da cui ordinare macchina nuova!
# HC_i_0_clienti_iniziali = 50 # quello con cui hai fatto la maggioranza delle simulazioni


# parametri/costanti
scala_lavoratori_capital_good_parametro_mail_roventini = 1#4 # Con 1 è come se non ci fosse
parametro_mail_roventini_scala_lavoratori = 1#4 # questa è la parte costante, sarà moltiplicata per la produttività media, di chi? Forse delle consumption
produttivita_economia_che_uso_in_scala_lav_rov = average_labor_productivity_iniziale # condizione iniziale poichè al turno=0 non calcolo la produttività media
theta_parameter_desired_inventories = 0.1
u_desired_level_capacity_utilization = 0.75
alpha_trigger_rule = 0.1
psi_2_quella_salario = 1
phi_wage_share_sussidio_rispetto_salario = 0.33
eta_max_machine_age = 19
b_payback_period = 8
lambda_max_debt_sale_ratio = 2
chi_replicator_dynamics_coefficient = -0.5
omega_1_competitiveness_weight = 1
omega_2_competitiveness_weight = 1
k_sample_coeff_nuovi_clienti = 0.01
# i_meno_1_quello_di_A_i_tau = -0.5 # nel paper sono uguali, io comunque li ho differenziati
# i_meno_2_quello_di_B_i_tau = -0.5
# i_piu_1_quello_di_A_i_tau = 0.5
# i_piu_2_quello_di_B_i_tau = 0.5
# sembra che -+0.5 fornisca numeri troppo grandi, forse siccome Windows opera con 32 bit anzichè 64 come Apple. Ad un certo punto fornisce errore nel turno=262 dicendo che int è troppo grande da convertire, ecco l'errore:
#" File "<ipython-input-4-4a20235d9da7>", line 1022, in <module> | D_j_domanda_turno_precedente[azienda_j_consumption] = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] # ora che non mi serve più aggiorno per il prossimo turno | OverflowError: int too big to convert"
i_meno_1_quello_di_A_i_tau = -0.15 # nel paper sono uguali, io comunque li ho differenziati
i_meno_2_quello_di_B_i_tau = -0.15
i_piu_1_quello_di_A_i_tau = 0.15
i_piu_2_quello_di_B_i_tau = 0.15


# sample T e durata simulazione
econometric_sample_size_T = 300-150 # quello per la simulazione seria, ma secondo me è esagerato anche perchè la produttività, cioè gli A e B, diverge
periodi_durata_simulazione_turni = 4 * econometric_sample_size_T
# periodi_durata_simulazione_turni = 70 #200
consumption_da_seguire_e_stampare = 10 # 2
#%%
# INIZIALIZZAZIONE
aziende_consumption_good = [average_labor_productivity_iniziale * np.ones(K_j_0_capital_stock_iniziale_per_azienda_beni) for ci in range(F1_numero_consumption_industry)] # è il vettore con i Theta_j_(t)  # Tecnicamente, è una lista di alcuni np.array
eta_macchinari_vecchiaia_consumption_aziende = [np.zeros(K_j_0_capital_stock_iniziale_per_azienda_beni, dtype=int) for _ in range(F1_numero_consumption_industry)] # è una lista di un numero F1 di np.array, come la lista sopra
numero_macchinari_K_intera_economia_storico_tutti_turni = np.array([])
AB_average_labor_productivity_tutta_economia = np.array([average_labor_productivity_iniziale]) # è uno "storico" come i vettori sotto, ma non è una lista poichè c'è una average a turno quindi basta un np.array, non serve una lista
# N.B.: In realtà quella del paper che al t=0 è 1 è quella totale AB e non quella A come hai supposto sopra per cui ti sei poi dovuto inventare anche la B che hai infatti posto uguale ad A. Semplicemente AB è la somma di tutti gli A_i_tau e i B_i_tau divisa per il numero totale di questi
A_average_productivity_solo_tutte_consumption_macchine = np.array([average_labor_productivity_iniziale])
A_average_productivity_solo_tutte_capital = np.array([average_labor_productivity_iniziale])
B_average_productivity_solo_tutte_capital = np.array([average_firm_labor_productivity_iniziale_supposta_da_me])

#clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità = [np.array([4-1,1-1],int), np.array([3-1,1-1],int), np.array([2-1],int)] # che visto come matrice_F2xF1_capital_puntano_a_consumption=np.array([ [1,0,0,1], [1,0,1,0], [0,1,0,0] ], int)
clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità = [ np.random.randint(0, F1_numero_consumption_industry, HC_i_0_clienti_iniziali) for ci in range(F2_numero_capital_industry) ] # sono le righe della matrice F2xF1 solo che la stessa azienda di consumption j (cioè F1) potrebbe essere estratta più volte per cui servirà un loop per correggere questo
# Però non è una lista "storico", semplicemente ad ogni turno aggiunge i nuovi clienti che sono in numero int( NC_i(t) ) quindi anche 0. La prossima lista di vettori è la versione speculare di questo, poi la matrice di adiacenza diretta riunisce entrambe le concezioni.
brochure_arrivate_alle_consumption_firms_con_indici_capital_firms = [ np.array([],int) for ci in range(F1_numero_consumption_industry) ] # sono F1 np.array che contengono gli indici _i delle capital firms alle quali le _j consumption firms mandano le brochure, NON sono i clienti storici HC_i(t). Sono le colonne della matrice.
# Non è uno "storico", ad ogni turno aggiunge le capital firms che hanno avuto come nuovi clienti quelle consumption firms.
# N.B.: ATTENTO! CHE A DIVERSE CONSUMPTION FIRM POTREBBE NON ESSERE ARRIVATA NEMMENO UNA BROCHURE E QUINDI NON HANNO NESSUNO DA CUI ORDINARE NUOVE MACCHINE A MENO CHE HC_i_0_clienti_iniziali non sia un numero abbastanza alto !!!
matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe = np.zeros((F2_numero_capital_industry, F1_numero_consumption_industry), dtype=int) # praticamente è una matrice di adiacenza di un grafo diretto
for righe_matrice_o_F2 in range(len(clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità)):
    # in realtà è una complicazione, bastava fare "for righe_matrice_o_F2 in range(F2_numero_capital_industry):" poichè quello è il numero di np.array nella lista
    contatore_confronto = 0 # serve per dopo per tenere conto della posizione degli elementi del for su ogni np.array della lista poichè sarà necessario aggiornare l'indice se è già stato estratto 
    for azienda_consumption_puntata in clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[righe_matrice_o_F2]:
        confronto = azienda_consumption_puntata
        while True:
            if matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[righe_matrice_o_F2, azienda_consumption_puntata] == 0:
                matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[righe_matrice_o_F2, azienda_consumption_puntata] = 1
                brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_consumption_puntata] = np.append(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_consumption_puntata], righe_matrice_o_F2)
                # così formo la lista (prima vuota) delle capital che hanno mandato una brochure alla consumption (la posizione dell'array nella lista) aggiungendo ogni volta una capital tra le F2 ad un np.array diverso delle consumption F1. Alla fine sono le colonne della matrice F2xF1 senza zeri però.
                if confronto != azienda_consumption_puntata:
                    clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[righe_matrice_o_F2][contatore_confronto] = azienda_consumption_puntata # serve poichè l'azienda potrebbe essere un duplicato nella estrazione dell'array sopra e quindi bisogna aggiornare l'array con l'azienda non duplicato
                contatore_confronto = 1 + contatore_confronto # serve a tenere traccia della posizione azienda_consumption_puntata all'interno dell'np.array per poterlo poi cambiare
                break
            else:
                azienda_consumption_puntata = np.random.randint(F1_numero_consumption_industry) # se c'è già un 1 in quel posto estraggo una nuova azienda consumption j che è puntata dalla stessa azienda di consumo i e ripeto il ciclo dove ora questa sarà la nuova
valore_HC_i_t_decimale_numero_clienti_di_ogni_i = HC_i_0_clienti_iniziali * np.ones(F2_numero_capital_industry) # serve poichè dobbiamo aumentarlo ogni turno aggiungendo al valore dell'azienda _i il NC_i(t)          
# non è uno storico con una lista che contiene un np.array per ogni turno, ma è un unico np.array che contiene solo il turno precedente e deve essere sovrascritto ogni turno
D_j_domanda_turno_precedente = np.round( (1+alpha_trigger_rule) * K_j_0_capital_stock_iniziale_per_azienda_beni * average_labor_productivity_iniziale * (u_desired_level_capacity_utilization/(1+theta_parameter_desired_inventories)) * np.ones(F1_numero_consumption_industry) )
# D_j_domanda_turno_precedente = np.round( (1+alpha_trigger_rule) * K_j_0_capital_stock_iniziale_per_azienda_beni * average_labor_productivity_iniziale * (1/(1+theta_parameter_desired_inventories)) * np.ones(F1_numero_consumption_industry) ) # senza u quindi 100% invece del 75%
# Se vuoi non fare partire tutte le aziende sincronizzate con la stessa domanda prevista ma invece sfasate!
#sfasamento_array = np.random.uniform(i_meno_1_quello_di_A_i_tau, i_piu_1_quello_di_A_i_tau, F1_numero_consumption_industry)
#D_j_domanda_turno_precedente = np.int64( np.around( ( (1/F1_numero_consumption_industry) * w_0_market_wage * (L_0_labor_supply_popolazione_lavoratori) * (1 + sfasamento_array) ) /( (1 + mi_0_mark_up_iniziale)*(w_0_market_wage/average_labor_productivity_iniziale) ) ) )
#for posizione_sfaso_in_array in range(len(sfasamento_array)):
#    # ovvero si va da 0 a F1, escluso. Devo togliere le D_exp_j_0 iniziali negative per non avere problemi
#    if D_j_domanda_turno_precedente[posizione_sfaso_in_array] < 0:
#        D_j_domanda_turno_precedente[posizione_sfaso_in_array] = np.around( ( (1/F1_numero_consumption_industry) * w_0_market_wage * (L_0_labor_supply_popolazione_lavoratori) ) /( (1 + mi_0_mark_up_iniziale)*(w_0_market_wage/average_labor_productivity_iniziale) ) )
#D_j_domanda_turno_precedente = np.int64( np.round( (1+alpha_trigger_rule) * K_j_0_capital_stock_iniziale_per_azienda_beni * average_labor_productivity_iniziale * (u_desired_level_capacity_utilization/(1+theta_parameter_desired_inventories)) * np.ones(F1_numero_consumption_industry) ) )
# Purtroppo, questo sopra non va bene e sembra che debba essere un numero decimale e non un int poichè anche con int64 dà problemi di overflow e quando ne fai la np.sum e lo metti nello storico tutti i turni di D_exp può venire un numero negativo senza motivo.
# Credo che l'overflow sia un problema di questo tipo: immagina che ogni variabile abbia abbastanza spazio per assumere in memoria un numero intero da -100 a +100, se tu volessi salvare 104, forse lo salverebbe come -97 poichè arrivato a +100, che è il suo massimo, va avanti a contare riiniziando dal -100. Penso sia qualcosa del genere.
# Questo sotto invece non va bene, nemmeno se lo si mette come np.int_( _, dtype=np.int64 ) poichè il tipo "int" di Python non è lo stesso di np.int32 ed è adatto a qualsiasi numero, invece int32 può contenere solo numeri da - a +2 miliardi circa e Windows di default trasforma qualsiasi int_ in int32, int64 invece può contenere miliardi di miliardi:
# D_j_domanda_turno_precedente = np.zeros(F1_numero_consumption_industry, dtype='int64') # lo inizializzo già così per evitare i bug con np.int_ probabilmente però astava mettere dtype='int64' in np.ones(F1_numero_consumption_industry) quindi: np.ones(F1_numero_consumption_industry, dtype='int64')
Q_j_quantita_prodotta_turno_precedente = np.zeros(F1_numero_consumption_industry) # non è uno storico con una lista che contiene un np.array per ogni turno, ma è un unico np.array che contiene solo il turno precedente e deve essere sovrascritto ogni turno
Q_j_t_quantita_prodotta_nel_turno_attuale = np.zeros(F1_numero_consumption_industry) # non è uno storico
S_j_turno_precedente_total_sales_soldi = np.zeros(F1_numero_consumption_industry) # non è uno storico con una lista che contiene un np.array per ogni turno, ma è un unico np.array che contiene solo il turno precedente e deve essere sovrascritto ogni turno
S_j_t_attuale_total_sales_soldi = np.zeros(F1_numero_consumption_industry)
prezzi_tutte_consumption_firms_p_j_t = (1 + mi_0_mark_up_iniziale)*(w_0_market_wage/average_labor_productivity_iniziale) * np.ones(F1_numero_consumption_industry) # non è uno storico con una lista che contiene un np.array per ogni turno, ma è un unico np.array che contiene solo il turno ATTUALE e deve essere sovrascritto ogni turno
Deb_j_turno_precedente_debito_consumption = np.zeros(F1_numero_consumption_industry)  # non è uno storico con una lista che contiene un np.array per ogni turno, ma è un unico np.array che contiene solo il turno precedente e deve essere sovrascritto ogni turno
# NW_j_t_liquid_assets_consumption = NW_i_j_0_liquid_assets * np.ones(F1_numero_consumption_industry) # non uno storico, ma un singolo np.array da sovrascrivere
# NW_i_t_liquid_assets_capital = NW_i_j_0_liquid_assets * np.ones(F2_numero_capital_industry) # non uno storico, ma un singolo np.array da sovrascrivere
NW_j_t_liquid_assets_consumption = scala_lavoratori_capital_good_parametro_mail_roventini * NW_i_j_0_liquid_assets * np.ones(F1_numero_consumption_industry) # devi aumentare anche i fondi iniziali dello stesso fattore per cui si troveranno a pagare di più
NW_i_t_liquid_assets_capital = scala_lavoratori_capital_good_parametro_mail_roventini * NW_i_j_0_liquid_assets * np.ones(F2_numero_capital_industry) # devi aumentare anche i fondi iniziali dello stesso fattore per cui si troveranno a pagare di più
# NW_j_t_liquid_assets_consumption = parametro_mail_roventini_scala_lavoratori * NW_i_j_0_liquid_assets * np.ones(F1_numero_consumption_industry) # devi aumentare anche i fondi iniziali dello stesso fattore per cui si troveranno a pagare di più
# NW_i_t_liquid_assets_capital = parametro_mail_roventini_scala_lavoratori * NW_i_j_0_liquid_assets * np.ones(F2_numero_capital_industry) # devi aumentare anche i fondi iniziali dello stesso fattore per cui si troveranno a pagare di più
quante_fallite_j_aziende_questo_turno_storico_tutti_turni = np.array([0], int)
quante_fallite_i_aziende_questo_turno_storico_tutti_turni = np.array([0], int)
contatore_restart_fallite_consumption_storico_tutti_turni = 0
contatore_restart_fallite_capital_storico_tutti_turni = 0
restart_j_t_fallite_tutte = 0
restart_i_t_fallite_tutte = 0 # forse non serve, è probabile che nel programma non verrà utilizzato
passato_solo_turno_dal_restart_mark_up = 0
numero_fallimenti_ogni_consumption_j = np.zeros(F1_numero_consumption_industry, dtype=int) # da questo np.array, alla fine della simulazione, puoi cercare se un indice è a zero e quindi una azienda _j eroica che è sopravvissuta per tutta la simulazione, una sorta di multinazionale, e vedere il suo monopolio come quota di mercato f_j
numero_fallimenti_ogni_capital_i = np.zeros(F2_numero_capital_industry, dtype=int)
D_expect_j_o_domanda_turno_precedente_storico_tutti_turni = np.array([])

# Attento che gli array che hanno soldi $ perciò quantità con la virgola devono essere inizializzati tali con 0.0 altrimenti, se lo fai con 0, l'array diventa un "int" e quando 0 verrà modificato ad esempio in 3.45, diventa invece automaticamente 3. Altra cosa è se invece aggiungi all'array un nuovo numero con la virgola tramite append 0.0 allora tutto l'array diventa decimale!
I_t_quantita_investment_consumption_firms_storico_tutti_turni = np.array([.0])
I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni = np.array([0.0])
RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni = np.array([.0])
RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni = np.array([0.0])
EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni = np.array([.0])
EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni = np.array([0.0])
Emp_occupazione_storico_tutti_turni = np.array([], int)
occupati_nelle_consumption_storico_tutti_turni = np.array([])
occupati_nelle_capital_storico_tutti_turni = np.array([])
produzione_massima_tutte_consumption_storico_tutti_turni = np.array([])
produzione_effettiva_tutte_consumption_storico_tutti_turni = np.array([])
D_N_variazione_magazzini_quantita_storico_tutti_turni = np.array([])
D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
C_consumi_quantita_storico_tutti_turni = np.array([])
C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
Y_pil_quantita_storico_tutti_turni = np.array([])
Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
# Quantità variazioni percentuali, hanno un elemento in meno degli altri vettori poichè sono differenze e non puoi fare differenza col turno 0-1. E' simile a fare le First Difference per il Dickey-Fuller Test, ma è una percentuale
D_N_variazione_percentuale_magazzini_quantita_storico_tutti_turni = np.array([])
D_N_variazione_percentuale_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
C_variazione_percentuale_consumi_quantita_storico_tutti_turni = np.array([])
C_variazione_percentuale_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
I_variazione_percentuale_investimenti_quantita_storico_tutti_turni = np.array([])
I_variazione_percentuale_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
Y_variazione_percentuale_pil_quantita_storico_tutti_turni = np.array([])
Y_variazione_percentuale_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
Emp_variazione_percentuale_occupazione_storico_tutti_turni = np.array([])
RI_variazione_percentuale_replacement_quantita_storico_tutti_turni = np.array([])
RI_variazione_percentuale_replacement_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
EI_variazione_percentuale_expansion_quantita_storico_tutti_turni = np.array([])
EI_variazione_percentuale_expansion_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
# Quantità variazioni, ma NON percentuali, hanno un elemento in meno degli altri vettori poichè sono differenze e non puoi fare differenza col turno 0-1. Sono le First Difference per il Dickey-Fuller Test
D_N_differenza_magazzini_quantita_storico_tutti_turni = np.array([]) # le quantità erano tutte dtype=int64 una volta poi tolto perchè potrebbe dare l'errore too big
D_N_differenza_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
C_differenza_consumi_quantita_storico_tutti_turni = np.array([])
C_differenza_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
I_differenza_investimenti_quantita_storico_tutti_turni = np.array([])
I_differenza_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
Y_differenza_pil_quantita_storico_tutti_turni = np.array([])
Y_differenza_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
Emp_differenza_occupazione_storico_tutti_turni = np.array([])
RI_differenza_replacement_quantita_storico_tutti_turni = np.array([])
RI_differenza_replacement_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])
EI_differenza_expansion_quantita_storico_tutti_turni = np.array([])
EI_differenza_expansion_soldi_quantita_x_prezzi_storico_tutti_turni = np.array([])

mi_mark_up_tutte_consumption_firms_storico_tutti_turni = [ mi_0_mark_up_iniziale * np.ones(F1_numero_consumption_industry) ]
# mi_mark_up e il prossimo vettore potrebbero essere se vuoi delle matrici dove le colonne sono F_1 le varie _j e le righe sono i turni,
# però hai preferito fare una lista di np.array, per ora hai creato il turno zero con F_1 elementi poi man mano dovrai aggiungere i vari turni che sono dei nuovi np.array in pratica. Per accedere all'azienda in un turno basta fare mi_mark_up[il turno][la j] 
mi_mark_up_tutte_consumption_firms_storico_tutti_turni.append(mi_0_mark_up_iniziale * np.ones(F1_numero_consumption_industry)) # POICHE' IO CREDO CHE IL MARK-UP AL turno=1 DEBBA ESSERE LO STESSO DEL turno=0 SICCOME NON POTRESTI CALCOLARLO POICHE' SERVIREBBE f_j(t-2) OVVERO QUELLA CORRISPONDENTE AL turno=-1 CHE OVVIAMENTE NON PUO' ESISTERE
f_market_share_tutte_consumption_firms_storico_tutti_turni = [ (1/F1_numero_consumption_industry) * np.ones(F1_numero_consumption_industry) ]
E_competitiveness_tutte_consumption_firms_t = np.zeros(F1_numero_consumption_industry) # non è uno "storico" con una lista che contiene un np.array per ogni turno, ma è un unico np.array che contiene solo il turno precedente e deve essere sovrascritto ogni turno
E_average_competitiveness_consumption_storico_tutti_turni = np.array([])
N_j_turno_precedente_merce_invenduta = np.zeros(F1_numero_consumption_industry)
w_t_salario_nominal_wage = np.array([w_0_market_wage]) # è uno storico come i vettori sopra che hanno "_storico" nel loro nome, ma non è una lista poichè c'è una average a turno quindi basta un np.array, non serve una lista
# oppure se preferisci che sia una lista: [ w_0_market_wage ]
aziende_capital_good_A_i_tau = average_labor_productivity_iniziale * np.ones(F2_numero_capital_industry)
aziende_capital_good_B_i_tau = average_firm_labor_productivity_iniziale_supposta_da_me * np.ones(F2_numero_capital_industry)
fake_innovazione_aziende_capital_good_A_i_tau = average_labor_productivity_iniziale * np.ones(F2_numero_capital_industry)
fake_innovazione_aziende_capital_good_B_i_tau = average_firm_labor_productivity_iniziale_supposta_da_me * np.ones(F2_numero_capital_industry)
prezzi_tutte_capital_firms_per_tecnologia_A_i_tau = ( w_0_market_wage/average_firm_labor_productivity_iniziale_supposta_da_me ) * np.ones(F2_numero_capital_industry)
#%%
# MAIN. ABM REGOLE
ora_dinizio = time.time()

for turno in range(periodi_durata_simulazione_turni):
    if restart_j_t_fallite_tutte == 1:
        # se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente bisogna cambiare le merci nei magazzini prima di copiare il vettore per i calcoli alla fine del turno
        N_j_turno_precedente_merce_invenduta = np.zeros(F1_numero_consumption_industry)
    N_j_copiato_per_calcolare_variazione_turno_precedente_magazzini = np.copy(N_j_turno_precedente_merce_invenduta)
    RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms = [ np.array([],int) for ci in range(F1_numero_consumption_industry) ]
    # è una lista di F1 np.array vuoti che conterranno gli indici [ovvero le posizioni delle A nell'np.array corrispondente in aziende_consumption_good] delle macchine A_i_tau che appartengono a Theta_j(t) di ogni consumption firm
    # che andranno sostituite e fanno parte delle RS_j(t), la loro quantità ovvero la len() di ogni np.array è RI_j(t).
    indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms = [ np.array([],int) for ci in range(F1_numero_consumption_industry) ]
    # è una lista di F1 np.array vuoti che conterranno gli indici [ovvero le posizioni delle A nell'np.array aziende_capital_good_A_i_tau] delle capital firms
    # che hanno vinto la gara come migliore A e prezzo per le A, appartenenti a Theta_j(t), da sostituire nelle RS_j(t). La posizione di tali A è nel vettore corrispondente (stessa len() di questo) di RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms.
    # numero macchine per replacement RI ordinate alla capital _i (RIGA) dalla consumption _j (COLONNA) 
    RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j = np.zeros((F2_numero_capital_industry, F1_numero_consumption_industry), dtype=int) # è una matrice
    # numero macchine in totale (RI+EI), per replacement e expansion, ordinate alla capital _i (RIGA) dalla consumption _j (COLONNA) 
    quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j = np.zeros((F2_numero_capital_industry, F1_numero_consumption_industry), dtype=int) # è una matrice
    EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare = np.zeros(F1_numero_consumption_industry, int) # è un np.array in cui per ogni azienda _j va inserito EI_j(t) ovvero il numero di nuove macchine A_i^tau TUTTE UGUALI che _j vuole comprare dalla STESSA azienda _i di beni capitali. Dopo dovrà essere aggiornato per ogni EI_j
    EI_indici_capital_firms_con_cui_espandere_theta_in_consumption_firms = (2+F2_numero_capital_industry) * np.ones(F1_numero_consumption_industry, int) # è un np.array della stessa lunghezza dell'np.array sopra che se il corrispondente EI_j è >0 deve dire l'indice della azienda capital _i da cui comprare tutte le macchine identiche A_i in quantità EI.
    # Siccome non deve essere usato se EI=0 è meglio che di default sia inizializzato ad un intero > F2 cioè più grande di quante capital ci sono in tutto così poi se andassi per errore a cercare quell'indice non lo troveresti!
    numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption = np.zeros(F2_numero_capital_industry, int) # è un np.array in cui per ogni azienda _i va inserito il numero totale (EI+RI) di nuove macchine A_i^tau TUTTE UGUALI che le varie diverse _j vogliono comprare. Dopo dovrà essere aggiornato per ogni EI_j e RI_j. # FORSE RIDONDANTE E SI POTRA' NON USARE E CANCELLARE!
    persone_ancora_disoccupate = L_0_labor_supply_popolazione_lavoratori # siccome dobbiamo sapere quante persone ancora sono libere per essere assunte perchè non è possibile che gli occupati superino la popolazione totale
    occupati_nelle_consumption = .0
    occupati_nelle_capital = .0
    if turno == 0:
        w_salario_questo_turno = w_0_market_wage
    else:
        # aggiorno i 6 np.array per gli investimenti aggiungendo un elemento 0 all'array che poi verrà aggiornato nei loop in seguito
        I_t_quantita_investment_consumption_firms_storico_tutti_turni = np.append(I_t_quantita_investment_consumption_firms_storico_tutti_turni, 0)
        I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni = np.append(I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni, 0)
        RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni = np.append(RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni, 0)
        RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni = np.append(RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni, 0)
        EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni = np.append(EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni, 0)
        EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni = np.append(EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni, 0)
        # qua va bene aggiungere 0 per quelli dei valori $ poichè l'array è stato inizializzato già all'inizio come un array con la virgola
        quante_fallite_j_aziende_questo_turno_storico_tutti_turni = np.append(quante_fallite_j_aziende_questo_turno_storico_tutti_turni, 0)
        quante_fallite_i_aziende_questo_turno_storico_tutti_turni = np.append(quante_fallite_i_aziende_questo_turno_storico_tutti_turni, 0)
        # mi_mark_up_tutte_consumption_firms_storico_tutti_turni.append(np.array([])) # dove questo "append" è il comando per le liste che è diverso da quello per aggiungere ai numpy array "np.append"
        if turno > 1:
            # poichè al turno=0,1 li avvei già creati sopra all'inizio che sono uguali
            mi_mark_up_tutte_consumption_firms_storico_tutti_turni.append(np.array([])) # dove questo "append" è il comando per le liste che è diverso da quello per aggiungere ai numpy array "np.append"
        # qua aggiungiamo un nuovo np.array per ora vuoto, che verrà riempito in seguito, per questo turno appena iniziato. In generale ogni array della lista mark-up è un turno, ogni turno contiene un valore per ogni azienda j, e stessa cosa per la quota di mercato f_j(t)
        f_market_share_tutte_consumption_firms_storico_tutti_turni.append(np.array([])) # il nuovo da riempire per questo turno, come sopra
        if restart_j_t_fallite_tutte == 1:
            # se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente e f_j(t) non verrà calcolata in questo turno qundi l'np.array rimarrà vuoto e allora bisogna inizializzarlo manualmente sennò poi al prossimo turno non si avrà f_j(t-1)
            f_market_share_tutte_consumption_firms_storico_tutti_turni[turno] = (1/F1_numero_consumption_industry) * np.ones(F1_numero_consumption_industry)
        AB_di_questo_turno_appena_calcolata = ( np.sum( np.sum(aziende_consumption_good[falso]) for falso in range(F1_numero_consumption_industry) ) + np.sum(aziende_capital_good_B_i_tau) )/( np.sum(len(fake) for fake in aziende_consumption_good) + len(aziende_capital_good_B_i_tau) ) # affinchè funzioni devi iterare i 2 loop come fai ora qua
        # Questa sotto invece, che era come avevi scrito prima, non funzionava poichè np.sum non può sommare tra loro np.array di una lista se hanno lunghezze diverse, all'inizio non capita perchè ciascuna _j ha 800 macchine A, però, nei turni successivi, accade siccome alcune ne comprano di più, altre di meno e il numero di ciascuna cambia.
        # Infatti questa non fa 2 loop come pensavi, ovvero prima fa la somma di ogni np.array e poi fa la somma dei risultati che sono, ciascuno, un intero; ciò che invece fa è provare a sommare ogni np.array e può farlo solo se hanno tutti la stessa lunghezza e siccome hanno la stessa lunghezza solo all'inizio, in generale fallisce:
        # AB_di_questo_turno_appena_calcolata = ( np.sum(aziende_consumption_good) + np.sum(aziende_capital_good_B_i_tau) )/( np.sum(len(fake) for fake in aziende_consumption_good) + len(aziende_capital_good_B_i_tau) )
        # ...+ np.sum(len(fake) for fake in aziende_capital_good_B_i_tau) ) non funziona poichè è un array e non una lista di array
        # Non viene usato aziende_capital_good_A_i_tau ma vengono usati il B e tutte le A delle aziende di consumo poichè nella economia rimangono le A di tau molto vecchie meno performanti delle ultime tecnologie che le aziende di capitale hanno nel vettore.
        # E questo riduce l'average AB rispetto a calcolarla solo sulle ultime A. Invece le B sono sempre le più nuove disponibili.
        # N.B.: ricorda che in generale dovrebbe essere diverso fare una media o fare la media delle medie
        # per cui hai fatto bene come qua sopra a fare tutto in un colpo solo sommando tutto e dividendo per il numero totale di macchine nell'economia
        # invece che fare prima la media delle A nei Theta_j(t) e poi fare la media dei B_i, sommarle e dividere per 2.
        # Infatti il problema è che se i 2 campioni hanno numerosità diversa (come qua) devi fare la media ponderata per dare lo stesso peso ai due campioni per cui non puoi semplicemente sommare e dividere per 2, che è sbagliato
        # con la media ponderata ti risalta fuori il risultato trovato sopra siccome ciascuna media è pesata per (numerosità quel campione)/(somma delle numerosità di tutti i campioni) e così non serve dividere per 2.
        #AB_di_questo_turno_appena_calcolata = ( np.sum(aziende_capital_good_A_i_tau) + np.sum(aziende_capital_good_B_i_tau) )/( len(aziende_capital_good_A_i_tau) + len(aziende_capital_good_B_i_tau) ) # variante: caso media solo capital sia A che B
        #AB_di_questo_turno_appena_calcolata = ( np.sum(aziende_capital_good_A_i_tau) )/( len(aziende_capital_good_A_i_tau) ) # variante: caso media solo capital solo A
        #AB_di_questo_turno_appena_calcolata = ( np.sum(aziende_capital_good_B_i_tau) )/( len(aziende_capital_good_B_i_tau) ) # variante: caso media solo capital solo B
        # AB_di_questo_turno_appena_calcolata = ( np.sum( np.sum(aziende_consumption_good[falso]) for falso in range(F1_numero_consumption_industry) ) )/( np.sum(len(fake) for fake in aziende_consumption_good) ) # variante: caso media solo consumption # usato per le simulazioni con scala_rov costante e variabile
        # variante: caso media consumption e capital sia A che B, tutto in pratica:
        #AB_di_questo_turno_appena_calcolata = ( np.sum( np.sum(aziende_consumption_good[falso]) for falso in range(F1_numero_consumption_industry) ) + np.sum(aziende_capital_good_A_i_tau) + np.sum(aziende_capital_good_B_i_tau) )/( np.sum(len(fake) for fake in aziende_consumption_good) + len(aziende_capital_good_A_i_tau) + len(aziende_capital_good_B_i_tau) )
        AB_average_labor_productivity_tutta_economia = np.append(AB_average_labor_productivity_tutta_economia, AB_di_questo_turno_appena_calcolata)
        w_salario_questo_turno = w_t_salario_nominal_wage[turno-1] * (1 + psi_2_quella_salario * (AB_average_labor_productivity_tutta_economia[turno] - AB_average_labor_productivity_tutta_economia[turno-1])/(AB_average_labor_productivity_tutta_economia[turno-1]) )
        # Anche se il salario dipendesse dalla percentuale di disoccupazione U=1-Emp(t)/L comunque sarebbe possibile calcolarlo poichè la produttività pi delle j serve sia per calcolare la domanda di lavoratori L^D_j(t) che il costo unitario c_j(t) ma le due sono slegate.
        # Però avresti bisogno della domanda di lavoratori anche delle i così dovresti fare due for per le j: nel primo calcoli L^D_j, poi fai il for sulle i calcolando L^D_i. Quindi puoi calcolare Emp(t) e w(t), infine riprendi le j con un altro for dove calcoli le c_j(t)=w(t)/pi_j(t) e finisci con le j calcolando le altre cose.
        w_t_salario_nominal_wage = np.append(w_t_salario_nominal_wage, w_salario_questo_turno) # aggiungo il salario di questo turno al vettore salari per usarlo come base per la Dw nel prossimo turno.
        # N.B.: Successivamente non dovrai usare w_t_salario_nominal_wage[turno] bensì w_salario_questo_turno poichè devi tenere conto del turno 0 che non era un vettore.
        # produttivita_economia_che_uso_in_scala_lav_rov = AB_di_questo_turno_appena_calcolata #1
        # produttivita_economia_che_uso_in_scala_lav_rov = ( np.sum(aziende_capital_good_B_i_tau) )/( len(aziende_capital_good_B_i_tau) ) # media delle B
        # scala_lavoratori_capital_good_parametro_mail_roventini = parametro_mail_roventini_scala_lavoratori * produttivita_economia_che_uso_in_scala_lav_rov # così in questa versione diventa una variabile composta da una costante che è il numero minimo di lavoratori per la produttività media scelta
        A_average_productivity_solo_tutte_consumption_macchine = np.append(A_average_productivity_solo_tutte_consumption_macchine, np.sum( np.sum(aziende_consumption_good[falso]) for falso in range(F1_numero_consumption_industry) ) / np.sum(len(fake) for fake in aziende_consumption_good) )
        A_average_productivity_solo_tutte_capital = np.append(A_average_productivity_solo_tutte_capital, np.sum(aziende_capital_good_A_i_tau) / len(aziende_capital_good_A_i_tau) )
        B_average_productivity_solo_tutte_capital = np.append(B_average_productivity_solo_tutte_capital, np.sum(aziende_capital_good_B_i_tau) / len(aziende_capital_good_B_i_tau) )
    # calcolo il nuovo prezzo delle tecnologia A_i_tau venduta dalle aziende capital _i
    # c_i_t_unit_cost_production = w_salario_questo_turno/aziende_capital_good_B_i_tau
    c_i_t_unit_cost_production = (w_salario_questo_turno/aziende_capital_good_B_i_tau) * scala_lavoratori_capital_good_parametro_mail_roventini
    # Mettendo questo calcolo qua, invece che dentro il loop dell capital firms, risolvi il problema che salvando i prezzi lo facevi col salario di questo turno anzichè correttamente con quello del prossimo
    prezzi_tutte_capital_firms_per_tecnologia_A_i_tau = np.copy(c_i_t_unit_cost_production) # Nel 2006 usava il mark-up anche per le capital _i, invece nel 2008 (ma nel successivo no!) dice espressamente "The price p_i is equal to the unit cost of production". Nel 2010 invece usa nuovamente un mark-up che è una costante.
    # Quindi, le capital firms _i non incassano nulla dalla vendita dei macchinari! Semplicemente ci ripagano il costo del lavoro (salario x #lavoratori occupati) che hanno anticipato SE IL SALARIO E' QUELLO CON CUI HANNO PRODOTTO E NON QUELLO DEL TURNO PRECEDENTE!
    # Nel paper del 2010 invece usa nuovamente un mark-up che è una costante!
    # mi_1_mark_up_capital_good_firms_rule=0.04
    # prezzi_tutte_capital_firms_per_tecnologia_A_i_tau = (1 + mi_1_mark_up_capital_good_firms_rule) * c_i_t_unit_cost_production

    # Se vuoi invece che seguano un ordine casuale così se c'è scarsità di lavoratori non accade che l'ultima azienda rimane sistematicamente fregata ad ogni turno poichè essendo sempre l'ultima della coda non può assumere siccome tutta la popolazione ha già un lavoro!
    # lista_consumption_da_rimescolare = [v for v in range(F1_numero_consumption_industry)]
    # np.random.shuffle(lista_consumption_da_rimescolare) # attento che non crea una nuova lista ma scombussola quella originale!
    # for azienda_i_capital in lista_consumption_da_rimescolare:
    for azienda_j_consumption in range(F1_numero_consumption_industry):
        if azienda_j_consumption == consumption_da_seguire_e_stampare:
            soldi_liquidi_j_prima_di_spenderli = NW_j_t_liquid_assets_consumption[azienda_j_consumption]#FORSE POI CANCELLARE
        if restart_j_t_fallite_tutte == 1:
            # se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente
            # qua metto quelle parti che sarebbe logico aspettarsi di trovarle nella parte che sta molto più sotto "Entry and Exit" delle consumption dove c'è cosa fare se sono tutte morte e i comandi per il restart.
            # Però non può andare là perchè non si possono alterare col restart le informazioni del turno in cui muoiono tutte siccome servono per raccogliere e calcolare le statistiche alla fine turno, così sei costretto a fare il restart delle variabili che trovi qua all'inizio del turno successivo, cioè ora!
            # Di questa, che è la domanda D_j(t-1), c'è sicuramente bisogno sennò l'azienda non produrrà niente questo turno!
            # Invece pare che la competitiveness E_j(t) non vada calcolata (stessa cosa del turno zero) per cui non ci sia bisogno di l_j(t) quantità: l_j_t_domanda_insoddisfatta = D_j_domanda_turno_precedente[azienda_j_consumption] - Q_j_quantita_prodotta_turno_precedente[azienda_j_consumption]
            # Ho assunto una domanda fittizia nel turno precedente e siccome questo non è il valore monetario $ della domanda (cioè quanto costa la somma di tutte le merci di consumo che i lavoratori comprano da una singola _j) bensì il numero di merci, devo dividere il valore monetario totale per il prezzo di un singolo bene di consumo di _j.
            D_j_domanda_turno_precedente[azienda_j_consumption] = np.round( (1+alpha_trigger_rule) * len(aziende_consumption_good[azienda_j_consumption]) * (np.sum(aziende_consumption_good[azienda_j_consumption])/len(aziende_consumption_good[azienda_j_consumption])) * (u_desired_level_capacity_utilization/(1+theta_parameter_desired_inventories)) ) # tolto int64 per evitare il problema dell'overflow che np.sum di D_exp dava un numero negativo
            Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] = 0
            #NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_i_j_0_liquid_assets
            # NW_j_t_liquid_assets_consumption[azienda_j_consumption] = scala_lavoratori_capital_good_parametro_mail_roventini * NW_i_j_0_liquid_assets * w_salario_questo_turno
            NW_j_t_liquid_assets_consumption[azienda_j_consumption] = parametro_mail_roventini_scala_lavoratori * NW_i_j_0_liquid_assets * w_salario_questo_turno
            # Queste due possono essere messe a zero tanto non vengono usate e viene saltato, come per il turno = 0, il computo di l_j. Precedentemente non eri sicuro che andassero messe a zero siccome al turno = 0 non le usavi:
            Q_j_quantita_prodotta_turno_precedente[azienda_j_consumption] = 0
            S_j_turno_precedente_total_sales_soldi[azienda_j_consumption] = 0
        D_expect_j_t = D_j_domanda_turno_precedente[azienda_j_consumption]
        N_desid_j_t_scorte_magazzino = theta_parameter_desired_inventories * D_expect_j_t
        Q_desid_j_t = D_expect_j_t + N_desid_j_t_scorte_magazzino - N_j_turno_precedente_merce_invenduta[azienda_j_consumption] # attento che la Q desiderata potrebbe venire negativa e dopo farà venire negativo anche il capitale K desiderato!
        if Q_desid_j_t < 0:
            Q_desid_j_t = 0
        K_j_t_macchine_turno_precedente = len(aziende_consumption_good[azienda_j_consumption]) # da quello che ho capito il capitale è semplicemente il numero di macchine dell'azienda j cioè la lunghezza del vettore
        
        pi_j_t_average_productivity = np.sum(aziende_consumption_good[azienda_j_consumption])/K_j_t_macchine_turno_precedente
        c_j_t_unit_cost_production = w_salario_questo_turno/pi_j_t_average_productivity
        # --- ORA, MIA SUPPOSIZIONE, POICHE' NEL PAPER NON DICE COSA E' Q_j(t) ---
        # -----
        Q_j_t_max_possibile_con_Theta_attuale = np.sum(aziende_consumption_good[azienda_j_consumption])
        if Q_desid_j_t < Q_j_t_max_possibile_con_Theta_attuale:
            Q_j_t_nel_paper_non_dice_come = np.round(Q_desid_j_t)
        else:
            Q_j_t_nel_paper_non_dice_come = np.round(Q_j_t_max_possibile_con_Theta_attuale)
        L_D_j_t_labor_demand = int( np.round(Q_j_t_nel_paper_non_dice_come/pi_j_t_average_productivity) )
        # -----
        # Come, invece, avevo supposto e scritto prima:
        # L_D_j_t_labor_demand = np.sum(aziende_consumption_good[azienda_j_consumption])/pi_j_t_average_productivity
        # Probabilmente, è sbagliato e la "production" Q_j(t) è la domanda al turno precedente D_e_j(t)=D_j(t-1) e quindi si potrebbe fare che se Sum A_i_tau di Theta_j => D_j(t-1) allora Q_j(t)=D_e_j(t), se altrimenti non riesci a produrre per soddisfare tale domanda (poichè Sum A_i_tau < D_j(t-1)) allora setti al max che puoi la produzione, cioè Q_j(t)=Sum A_i_tau PERO' mancano gli inventari. Oppure se...
        # oppure Q_j(t) è il livello desiderato di produzione Q_d_j(t), però mi sembra difficile poichè da quello si calcolano quante macchine ordinare in futuro...
        # oppure potrebbe essere una percentuale ignota della capacità totale, ad esempio il 75% * sum A_i_tau appartententi a Theta_j, una cosa analoga al u_d in K_d_j = Q_d_j(t)/u_d
        # Ma se invece è proprio così, allora L_D_j(t) = Sum A_j_tau / pi_j(t) = Sum A_j_tau / [Sum A_j_tau/K_j(t)] = K_j(t) quindi Capitale=Lavoratori o meglio #macchine=#lavoratori che ha senso siccome A_i_tau è la produttività di UN SINGOLO lavoratore e stiamo utilizzando tutta la capacità produttiva ovvero tutte le macchine, tutti gli A_i di Theta_j(t) !!!
        if L_D_j_t_labor_demand > persone_ancora_disoccupate:
            # siccome non si possono assumere più lavoratori di quanti ne siano disponibili nella popolazione tra i disoccupati
            L_D_j_t_labor_demand = persone_ancora_disoccupate
            persone_ancora_disoccupate = 0
            occupati_nelle_consumption += persone_ancora_disoccupate
            Q_j_t_nel_paper_non_dice_come = np.round(L_D_j_t_labor_demand * pi_j_t_average_productivity)
        else:
            persone_ancora_disoccupate = persone_ancora_disoccupate - L_D_j_t_labor_demand # aggiorno togliendo i lavoratori che ora sono stati occupati
            occupati_nelle_consumption += L_D_j_t_labor_demand
        if turno<2:
            # non vale solo per il turno=0 ma anche per il turno=1 siccome se dovessimo calcolare mi avremmo bisogno della market share f_j non solo al turno t-1 ma anche t-2 ma al turno=1 sarebbero f_j(0) e f_j(-1) che non le abbiamo perciò f_j(t-1) e f_j(t-2) per almeno t>=2.
            # Però al vettore dei mark-up delle aziende per i primi 2 turni cioè turno=0,1 ho messo il mark-up iniziale del turno=0 già alla creazione del vettore.
            mi_j_t_mark_up = mi_0_mark_up_iniziale
        else:
            if restart_j_t_fallite_tutte == 1 or passato_solo_turno_dal_restart_mark_up == 1:
                # se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente
                mi_j_t_mark_up = mi_0_mark_up_iniziale
            else:
                mi_j_t_mark_up = mi_mark_up_tutte_consumption_firms_storico_tutti_turni[turno-1][azienda_j_consumption] * (1 + (f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1][azienda_j_consumption] - f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-2][azienda_j_consumption])/(f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-2][azienda_j_consumption]) )
            mi_mark_up_tutte_consumption_firms_storico_tutti_turni[turno] = np.append(mi_mark_up_tutte_consumption_firms_storico_tutti_turni[turno], mi_j_t_mark_up) # al vettore np.array del turno (creato prima) aggiungo il valore di mi della azienda j
        p_j_t_prezzo_singola_merce_consumption_good = (1 + mi_j_t_mark_up)*c_j_t_unit_cost_production
        prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption] = p_j_t_prezzo_singola_merce_consumption_good # lo salvo per dopo poichè questo verrà cancellato alla fine del loop, ma a fine programma dovrò dividere il valore in soldi € della domanda rivolta alla _j per il suo prezzo e fare round() per sapere la quantità di merce che vende D_j(t) che servirà nel turno (t+1)
        if turno>0 and restart_j_t_fallite_tutte == 0:
        #if turno>0: # prima era così poi hai aggiunto la parte restart poichè
            # ora bisogna tenere conto anche se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente
            # dal turno=1 in poi credo che sia possibile calcolare la competitiveness E_j(t)
            l_j_t_domanda_insoddisfatta = D_j_domanda_turno_precedente[azienda_j_consumption] - Q_j_quantita_prodotta_turno_precedente[azienda_j_consumption]
            if l_j_t_domanda_insoddisfatta > 0:
                # questo è il caso in cui c'è domanda insoddisfatta poichè Q_j(t-1)<D_j(t-1) la quantità prodotta al turno prima era inferiore alla richiesta, l_j(t) è perciò positiva e col segno meno nella formula di E_j diventerà negativa riducendo la competitività: cioè un numero negativo ancora più grande
                E_j_t_competitiveness = - omega_1_competitiveness_weight*p_j_t_prezzo_singola_merce_consumption_good - omega_2_competitiveness_weight*l_j_t_domanda_insoddisfatta
            else:
                # invece questo è il caso dove Q_j(t-1)>=D_j(t-1) perciò l_j(t) come definita sopra sarebbe o negativa o nulla e poi nella formula verrebbe positiva e aumenterebbe la E_j, potrebbe anche renderla positiva!
                # Però nel paper la chiama "livello domanda insoddisfatta" quindi, SECONDO ME, io intendo che deve esserci nella formula solo se c'è domanda insoddisfatta, quindi in questo caso non deve comparire, l_j(t)=0 
                l_j_t_domanda_insoddisfatta = 0 # in realtà non importa che la metti a zero, tanto nella formula qua sotto non compare, è solo per chiarezza
                E_j_t_competitiveness = - omega_1_competitiveness_weight*p_j_t_prezzo_singola_merce_consumption_good
            # -----
            # Come, invece, avevo supposto e scritto prima:
            # l_j_t_domanda_insoddisfatta = D_j_domanda_turno_precedente[azienda_j_consumption] - Q_j_quantita_prodotta_turno_precedente[azienda_j_consumption] # D_j(t-1)-Q_j(t-1)<0 cioè Q è maggiore, l_j è negativo per cui poi nella formula di E_j diventa positivo perchè c'è il meno e riduce l'altro termine che è negativo, io credo che E_j sia negativa
            # E_j_t_competitiveness = - omega_1_competitiveness_weight*p_j_t_prezzo_singola_merce_consumption_good - omega_2_competitiveness_weight*l_j_t_domanda_insoddisfatta
            E_competitiveness_tutte_consumption_firms_t[azienda_j_consumption] = E_j_t_competitiveness # la metto nel vettore dove ci sono tutte, poichè alla fine avrò bisogno di tutte quante per fare la media
        costo_totale_della_produzione_c_j_t_Q_j_t = c_j_t_unit_cost_production * Q_j_t_nel_paper_non_dice_come
        if S_j_turno_precedente_total_sales_soldi[azienda_j_consumption] > 0.001:
            # poichè devo evitare la divisione per zero se S_j(t-1)=0 che è il caso del primo turno e di quando un'azienda _j muore e rinasce appunto senza aver venduto nulla prima
            # Però ricorda che il rapporto_debito_vendite_j_turno_precedente verrà zero anche con S_j>0 siccome se il debito D_j=0            
            rapporto_debito_vendite_j_turno_precedente = Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]/S_j_turno_precedente_total_sales_soldi[azienda_j_consumption]
        else:
            rapporto_debito_vendite_j_turno_precedente = 0
        debito_totale_j_con_nuovo_ora = Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] # poi sarà aggiornato
        quanto_debito_si_puo_ancora_fare = 0 # poi sarà aggiornato
        if rapporto_debito_vendite_j_turno_precedente < lambda_max_debt_sale_ratio:            
            # caso credito per j è NON è RATIONED PRIMA della produzione (a priori diciamo). Magari lo diventa (a posteriori) con la produzione o con la produzione + espansione o con produzione + espansione + replacement oppure j riesce a fare tutto!
            if rapporto_debito_vendite_j_turno_precedente == 0:
                # questo serve poichè all'inizio le _j hanno sia debito=sales=0 quindi ne hai tenuto conto sopra mettendo "rapporto_debito_vendite_j_turno_precedente=0" poichè sennò avrebbe diviso per zero siccome era 0/0. Inoltre hai bisogno di un rapporto deciso da te sennò se D_j=0 sarà sempre zero!
                quanto_debito_si_puo_ancora_fare = NW_i_j_0_liquid_assets # arbitrario, deciso da me. Assumo che all'inizio possa fare tanto debito quanto gli asset iniziali, perciò al primo turno si ritrova massimo a poter spendere 2*NW_0 o forse è troppo alto?! Meglio una metà o un quarto
            else:
                # Come era il programma prima. Questa era l'unica strada, era di default, e questo if non esisteva: però non funzionava siccome le aziende non si indebitavano mai poichè è un cane che si morde la coda, se il debito all'inizio è D_j=0 rimarrà per sempre zero!
                quanto_debito_si_puo_ancora_fare = S_j_turno_precedente_total_sales_soldi[azienda_j_consumption] * lambda_max_debt_sale_ratio - Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
            # che è parte della formula scritta solo nel paper del 2006:  c_j(t)Q_j(t) - NW_j(t-1) > S_j(t-1)Omega_max - Deb_j(t-1)  dove Deb_max(t-1) = S_j(t-1)Lambda. Omega_max e Lambda sono la stessa cosa, solo due notazioni diverse.
            if (costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]) > quanto_debito_si_puo_ancora_fare:
                # Allora j è RATIONED già solo per l'intera PRODUZIONE (ma si può indebitare per una parte di essa)! Non ha senso guardare espansione e replacement quindi puoi annullare i vettori sopra oppure puoi fare che vengono calcolati solo nel caso in cui sia "False" così risparmi tempo di calcolo.
                # ATTENTO che se quello di sopra è "False", poichè la differenza è negativa, allora significa che NW basta per finanziare la produzione! E, almeno per la produzione, non c'è bisogno di indebitarsi.
                quanto_mancherebbe_che_pero_non_puo_fare = costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption] - quanto_debito_si_puo_ancora_fare # bisogna quindi calcolare quanta produzione può permettersi con i suoi soli NW e il debito che manca per arrivare a Lambda
                produzione_permessa = costo_totale_della_produzione_c_j_t_Q_j_t - quanto_mancherebbe_che_pero_non_puo_fare # ovvero: + NW_j_t_liquid_assets_consumption[azienda_j_consumption] + quanto_debito_si_puo_ancora_fare
                # Oppure, più semplicemente:
                # produzione_permessa = NW_j_t_liquid_assets_consumption[azienda_j_consumption] + quanto_debito_si_puo_ancora_fare
                quantità_Q_permessa = produzione_permessa/c_j_t_unit_cost_production # produzione_permessa non è una quantità ma una c_jQ_j quindi per ottenere Q_j devi dividere per c_j
                lavoratori_L_effettivamente_permessi_da_assumere = quantità_Q_permessa/pi_j_t_average_productivity
                # siccome in realtà i lavoratori da assumere sono meno di quelli che si sarebbe voluto ma che abbiamo già sotratto dai disoccupati totali, avendo tolto troppo, dobbiamo riaggiungere ai disoccupati quelli che alla fine non sono stati assunti
                persone_ancora_disoccupate = persone_ancora_disoccupate + int( np.round(L_D_j_t_labor_demand - lavoratori_L_effettivamente_permessi_da_assumere) ) # poichè  L_D_j_t_labor_demand > lavoratori_L_effettivamente_permessi_da_assumere
                occupati_nelle_consumption -= int( np.round(L_D_j_t_labor_demand - lavoratori_L_effettivamente_permessi_da_assumere) )
                L_D_j_t_labor_demand = int( np.round(lavoratori_L_effettivamente_permessi_da_assumere) ) # aggiorno con un bagno di realtà
                Q_j_t_nel_paper_non_dice_come = np.round(quantità_Q_permessa)
                debito_totale_j_con_nuovo_ora = quanto_debito_si_puo_ancora_fare + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] # che dovrebbe essere uguale a  S_j_turno_precedente_total_sales_soldi[azienda_j_consumption]*lambda_max_debt_sale_ratio
                #costo_totale_della_produzione_c_j_t_Q_j_t = produzione_permessa # lo aggiorno poichè dopo se userò "costo_totale_della_produzione_c_j_t_Q_j_t" deve essere solo quella effettivamente fatta
                costo_totale_della_produzione_c_j_t_Q_j_t = c_j_t_unit_cost_production * Q_j_t_nel_paper_non_dice_come # lo aggiorno poichè dopo se userò "costo_totale_della_produzione_c_j_t_Q_j_t" deve essere solo quella effettivamente fatta
                EI_j_t_espansione_investimenti = 0 # poichè non riesci nemmeno ad assumere tutte i lavoratori per produrre la quantità che volevi figuriamoci comprare le macchine
                RI_j_t_replacement_investimenti = 0 # pero' cosi' le macchine piu' vecchie di eta saranno buttate via e non sostituite e cosi' il # delle A in Theta_j calera'
                prezzo_tutte_nuove_A_in_quantita_EI = 0
                prezzo_tutte_replacement_macchine_A_RI = 0
                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = (NW_j_t_liquid_assets_consumption[azienda_j_consumption] + quanto_debito_si_puo_ancora_fare) - costo_totale_della_produzione_c_j_t_Q_j_t # devi consumare tutto il tuo NW (il cash), prima di indebitarti per il restante però siccome non puoi usare quantità non intere di lavoratori potrebbe essere < NW di fatto
            else:
                # PER ORA CONSIDERIAMO CASO CHE NUOVO DEBITO BASTA PER TUTTA E SOLA LA PRODUZIONE CIOE' CHE SIA ESATTAMENTE:
                # (costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]) == quanto_debito_si_puo_ancora_fare
                # ...e quindi  differenza_debito = 0
                differenza_debito = quanto_debito_si_puo_ancora_fare - (costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption])
                # sicuramente l'escamotage "differenza_debito" non è negativo, però non è detto sia >0, potrebbe anche essere =0 e quindi riesci a fare tutta e solo la produzione. Non lo sai ancora...
                # Comunque sia, in questo if va bene la quantità Q fissata sopra dove compare per la prima volta "Q_j_t_nel_paper_non_dice_come"
                if differenza_debito > 0:
                    # Allora j NON è RATIONED per l'INTERA produzione (la parte di essa mancante la copre col nuovo debito e poi rimane ancora qualcosina di debito disponibile per investimenti)!
                    # Ha senso guardare espansione e replacement. Potrebbe però esserlo per tutta la produzione + espansione oppure per produzione + espansione + replacement o j riesce a fare tutto!
                    # Quindi potresti spostare qua la parte delle brochure messa sopra e quella delle ricerca della A che minimizza l'equazione!
                    # -----
                    # -----
                    # HO ACCORPATO QUA ANCHE LA PARTE PER CALCOLARE "EI_j_t_espansione_investimenti" CHE ERA, PRIMA, INIZIALMENTE ALL'INIZIO DEL LOOP PER LE _j, SICCOME NON HA SENSO CALCOLARE QUANTE NUOVE MACHINE EI VORREBBERO SE POI NON LE POSSONO COMPRARE POICHE SONO rationed, SPRECO DI TEMPO!
                    # K_desid_j_t_capitale_auspicato = Q_desid_j_t/u_desired_level_capacity_utilization # come scrivono nel paper senza la produttività media
                    K_desid_j_t_capitale_auspicato = Q_desid_j_t/(pi_j_t_average_productivity * u_desired_level_capacity_utilization) # mia supposizione
                    K_trig_j_soglia_per_investire = K_j_t_macchine_turno_precedente * (1 + alpha_trigger_rule)
                    # FORSE QUESTA PARTE NON ANDRA' TENUTA NELLA VERSIONE DEL 2010
                    if K_desid_j_t_capitale_auspicato < np.round(K_trig_j_soglia_per_investire): # sennò quando sono uguali non funziona perchè ci sono sempre degli altri numeri che mette Python dopo tanti zeri, es. tu ti immagini sia 880.0<880.0: False, invece in realtà è 880.0<880.0000001: True !!!
                        EI_j_t_espansione_investimenti = 0
                    else:
                        EI_j_t_espansione_investimenti = int( np.round(K_trig_j_soglia_per_investire - K_j_t_macchine_turno_precedente) ) # per cui espande sempre e solo di alfa*K. Così è nel 2006 e nel 2008, cambierà invece nel paper successivo cioè il 2010
                    # STRANO che non sia K_desid_j_t_capitale_auspicato invece di K_trig_j_soglia_per_investire ed effettivamente nel 2010, al contrario del 2006 e 2008, è proprio così!!! Ovvero:
                        # EI_j_t_espansione_investimenti = int( np.round(K_desid_j_t_capitale_auspicato - K_j_t_macchine_turno_precedente) )
                    # simile a quella presente sotto più avanti, ma diversa poichè è per EI e non per RI
                    if len(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]) != 0:
                        # poichè esiste il caso che alla azienda consumption _j non sia arrivata nemmeno una brochure da nessuna capital _i
                        valore_eq_da_minimizzare = np.array([]) # alla fine del "for" seguente, questo vettore avrà la stessa dimensione del np.array brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption].
                        # Infatti qua valuti l'eq. per tutti non solo per quelle brochure che hanno superato la prova con la A app. Theta_j(t), poichè quella A è da buttare perchè vecchia
                        for indice_azienda_capital_brochure in brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]:
                            # questa è l'equazione nel paper 2010. In realtà sarebbe:  p*_i + b c_i  ma c'è il salario w(t) al numeratore di entrambe le frazioni per cui si può cancellare
                            eq_paper_2010_migliore_il_piu_piccolo = (1/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) + b_payback_period*(1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure])
                            # Nel caso che invece usi quello che dice nel 2006 a pag.7: "highest productivity/price ratio" anche se lascio come nome il più piccolo per non dover cambiare tutto anche se ora bisogna prendere il massimo e non il minimo. Ciò che hai pensato è che il bug sia qua.
                            # Infatti come era prima preferivi b volte di più quelli che hanno un alto A rispetto a quelli che hanno un alto B quindi due che hanno lo stesso valore come minimo dovrebbero avere uno la B b volte minore della A dell'altro.
                            # Per quello viene che la A media è sempre superiore della B media nel grafico poichè sistematicamente quelle che hanno estratto alto B e basso A muoiono poichè vengono preferiti quelli che hanno estratto un alto A e un basso B ben b volte di più!
                            #eq_paper_2010_migliore_il_piu_piccolo = aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]/prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[indice_azienda_capital_brochure]
                            valore_eq_da_minimizzare = np.append(valore_eq_da_minimizzare, eq_paper_2010_migliore_il_piu_piccolo)
                        # attento che se ce ne sono due o più con lo stesso minimo, fornisce la posizione nell'np.array (l'indice) solo del primo di essi
                        posizione_minimo = np.argmin(valore_eq_da_minimizzare)
                        #posizione_minimo = np.argmax(valore_eq_da_minimizzare) # se invece prendo il valore massimo poichè massimizzo il rapporto A/p ovvero B(A/w) ma w è lo stesso per tutte le _i quindi è una costante per cui AB/const diventa all'atto pratico che massimizzo la moltiplicazione AB
                        vincitore_indice_capital_azienda_nella_brochure = brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption][posizione_minimo]
                        EI_indici_capital_firms_con_cui_espandere_theta_in_consumption_firms[azienda_j_consumption] = vincitore_indice_capital_azienda_nella_brochure
                        # prezzo_di_una_sola_nuova_A_selezionata_per_EI = w_salario_questo_turno/aziende_capital_good_B_i_tau[vincitore_indice_capital_azienda_nella_brochure]
                        # OPPURE SE IL PREZZO DEI MACCHINARI A VIENE FISSATO CON ANCHE IL MARK-UP COME NEL 2010 mi_1=0.04 ALLORA SI DEVE USARE IL VETTORE PREZZI POICHE' NEL TURNO PRECEDENTE AVRAI MESSO TUTTI I PREZZI COMPRESI DI mi_1 LI'
                        prezzo_di_una_sola_nuova_A_selezionata_per_EI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] # nel caso 2010 e che ci sia parametro_scala_roventini per le capital
                        # prezzo_tutte_nuove_A_in_quantita_EI = int( np.round(EI_j_t_espansione_investimenti) ) * prezzo_di_una_sola_nuova_A_selezionata_per_EI
                        prezzo_tutte_nuove_A_in_quantita_EI = np.round(EI_j_t_espansione_investimenti) * prezzo_di_una_sola_nuova_A_selezionata_per_EI
                        # Attento che esiste il caso in cui EI_j_t_espansione_investimenti=0 per cui dovrai alla fine controllare se e diverso da zero prima di dare all'azienda _i vincitrice di produrre le macchine                    # -----
                    # -----
                        if differenza_debito > prezzo_tutte_nuove_A_in_quantita_EI:
                            numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure] = int( np.round(EI_j_t_espansione_investimenti) ) + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure]
                            quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = int( np.round(EI_j_t_espansione_investimenti) ) + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                            # -----
                            # -----
                            # -----
                            RI_j_t_replacement_investimenti = 0 # replacement investment poi verrà aggiornato
                            prezzo_tutte_replacement_macchine_A_RI = 0 # poi verrà aggiornato
                            while True:
                                for indice_A_posseduta_appartenente_theta_j_t in range(K_j_t_macchine_turno_precedente):
                                    A_posseduta_appartenente_theta_j_t = aziende_consumption_good[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t]
                                    #if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= eta_max_machine_age:
                                    # penso sia meglio mettere -1 o -2 all'età massima sennò le aziende hanno a disposizione un solo turno per sostituire tutte le macchine che a fine turno si romperanno!
                                    if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= (eta_max_machine_age-2):
                                        # Il vincitore lo conosci già dal calcolo fatto prima per EI che è lo stesso calcolo. Nota che il calcolo sopra viene fatto anche nel caso in cui il calcolo all'inizio del programma desse EI_j_t_espansione_investimenti=0 per cui in realtà non ci sarebbe bisogno di cercare macchine con cui espandere la produzione.
                                        # E' meglio fatto così poichè sopra la ricerca del vincitore viene fatta una volta sola per ogni consumption _i, invece qua sarebbe stata fatta per ogni A con età>=eta quindi ripetevi un sacco di volte quel calcolo per trovare sempre lo stesso vincitore, sarebbe stupido!!!
                                        # La differenza qua rispetto alla parte sopra per EI, sta nelle prossime 2 righe:
                                        if differenza_debito >= (prezzo_tutte_nuove_A_in_quantita_EI + prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] + prezzo_tutte_replacement_macchine_A_RI):
                                            RI_j_t_replacement_investimenti = RI_j_t_replacement_investimenti + 1 # aggiorno poichè la macchina è da cambiare perchè troppo vecchia e l'azienda j se la può permettere, non è rationed.
                                            prezzo_tutte_replacement_macchine_A_RI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] + prezzo_tutte_replacement_macchine_A_RI
                                            RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption] = np.append(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption], indice_A_posseduta_appartenente_theta_j_t)
                                            indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption] = np.append(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption], vincitore_indice_capital_azienda_nella_brochure)
                                            numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure] = 1 + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure]
                                            quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = 1 + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                                            RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = 1 + RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                                        else:
                                            break
                                    else:
                                        # Devi farlo così, con i loop di for poichè, anche se sembra una buona idea, non può funzionare fare i conti operando sui vettori in un solo colpo invece che su numeri singoli: questo a causa del fatto che stai usando indici che rimandano a posizioni diverse in quei vettori!
                                        indici_candidati_prova_payback_superata = np.array([], int)
                                        valore_rapporto_candidati_superata = np.array([])
                                        for indice_azienda_capital_brochure in brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]:
                                        # In definitiva anche se (1/B)/(1/A - 1/A_new) <= b scritta nel paper e 1/B <= b(1/A - 1/A_new) sembrano la stessa cosa, non lo sono quando 1/A - 1/A_new è negativo ovvero A>A_new fornisce un segno meno al denominatore.
                                        # Infatti in questa circostanza, l'equazione logicamente analoga come grado di verità a (1/B)/(1/A - 1/A_new) <= b de paper sarebbe (1/B) >= b(1/A - 1/A_new) che NON è l'equazione che ti serve nel modello!
                                        # Quindi quella del paper vale solo nel caso che il denominatore sia positivo ovvero A<A_new in quel caso moltiplicare per il denominatore per metterlo dall'altra parte e cancellarlo a sinistra non fornisce nessun problema,
                                        # nessun bisogno di cambiare il segno della disequazione poichè stai moltiplicando per un numero positivo. Invece (1/B) <= b(1/A - 1/A_new) funziona sempre, anche in caso di segno negativo se A>A_n infatti +<- fornisce "False" e quindi l'if viene saltato non essendo "True".
                                            if (1/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):
                                            # Invece questo è il caso in cui i prezzi dei macchinari sono parametro*salario/B poichè servono parametro lavoratori per fare B macchinari
                                            # if (scala_lavoratori_capital_good_parametro_mail_roventini/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):# investimenti non periodici perchè anche RI 2 turni consecutivi stesso macchinario
                                            # Questo è il caso in cui non si può sostituire una A prima di b turni sennò una RI potrebbe essere sostituita ogni turno e in cui i prezzi dei macchinari sono parametro*salario/B poichè servono parametro lavoratori per fare B macchinari:
                                            # if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= b_payback_period and (scala_lavoratori_capital_good_parametro_mail_roventini/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):# variabile
                                            #if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= b_payback_period and (parametro_mail_roventini_scala_lavoratori/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):# fisso
                                                indici_candidati_prova_payback_superata = np.append(indici_candidati_prova_payback_superata, indice_azienda_capital_brochure)
                                                minimizzare_il_migliore_il_piu_piccolo = abs( (1/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure])/( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ) )
                                                # Nel caso che invece usi quello che dice nel 2006 a pag.7: "highest productivity/price ratio" anche se lascio come nome il più piccolo per non dover cambiare tutto anche se ora bisogna prendere il massimo e non il minimo.
                                                #minimizzare_il_migliore_il_piu_piccolo = aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]/prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[indice_azienda_capital_brochure]
                                                valore_rapporto_candidati_superata = np.append(valore_rapporto_candidati_superata, minimizzare_il_migliore_il_piu_piccolo)
                                        if len(indici_candidati_prova_payback_superata) > 0:
                                            # ovvero se qualcuno ha vinto: infatti la A posseduta in Theta_j(t) potrebbe essere migliore di tutte le A delle brochure
                                            # attento che se ce ne sono due o più con lo stesso minimo, fornisce la posizione nell'np.array (l'indice) del primo di essi
                                            posizione_minimo = np.argmin(valore_rapporto_candidati_superata)
                                            #posizione_minimo = np.argmax(valore_rapporto_candidati_superata) # se invece prendo il valore massimo poichè massimizzo il rapporto A/p ovvero B(A/w) ma w è lo stesso per tutte le _i quindi è una costante per cui AB/const diventa all'atto pratico che massimizzo la moltiplicazione AB
                                            vincitore_indice_capital_azienda = indici_candidati_prova_payback_superata[posizione_minimo]
                                            if differenza_debito >= (prezzo_tutte_nuove_A_in_quantita_EI + prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda] + prezzo_tutte_replacement_macchine_A_RI):
                                                RI_j_t_replacement_investimenti = RI_j_t_replacement_investimenti + 1 # aggiorno poichè la macchina è da cambiare perchè tecnologicamente superata e l'azienda j se la può permettere, non è rationed.
                                                prezzo_tutte_replacement_macchine_A_RI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda] + prezzo_tutte_replacement_macchine_A_RI
                                                RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption] = np.append(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption], indice_A_posseduta_appartenente_theta_j_t)
                                                indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption] = np.append(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption], vincitore_indice_capital_azienda)
                                                numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda] = 1 + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda]
                                                quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption] = 1 + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption]
                                                RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption] = 1 + RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption]
                                            else:
                                                break
                            # -----
                            # -----
                            # -----
                                # SICCOME DEVE ESSERCI UNA INTERRUZIONE AL WHILE INFINITO poichè potrebbe essere che se le A di Theta_j sono giovani e sono le migliori non vengano mai incontrati i due break messi sopra!
                                # SECONDO ME DEVE STARE SULLA STESSA RIENTRANZA DEL "for indice_A_posseduta_appartenente_theta_j_t in range(K_j_t_macchine_turno_precedente):" POICHE' QUANDO è FINITO E SONO SCORSE TUTTE LE A di Theta_j DEVE INTERROMPERE IL WHILE!
                                break
                            # Qua in poi, ci troviamo sulle stesse righe (ovvero rientranza) di quelle sotto (cioè che seguono) "if differenza_debito > prezzo_tutte_nuove_A_in_quantita_EI:"
                            if (NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzo_tutte_replacement_macchine_A_RI - prezzo_tutte_nuove_A_in_quantita_EI - costo_totale_della_produzione_c_j_t_Q_j_t) <= 0:
                                debito_totale_j_con_nuovo_ora = (prezzo_tutte_replacement_macchine_A_RI + prezzo_tutte_nuove_A_in_quantita_EI + costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]) + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
                                # poichè abbiamo detto che la differenza sopra nelle parentesi dell'if è negativa o nulla quindi ho dovuto invertire i segni ovvero: -(..differenza dell'if..)
                                # c'è anche un altro modo meno ovvio e più incasinato per farlo:
                                # debito_totale_j_con_nuovo_ora = quanto_debito_si_puo_ancora_fare - (differenza_debito - prezzo_tutte_nuove_A_in_quantita_EI - prezzo_tutte_replacement_macchine_A_RI) + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
                                # in questa, esplicitando i calcoli e la definizione di differenza_debito, si cancella quanto_debito_si_puo_ancora_fare e viene esattamente la formula sopra, è solo un modo molto più complicato di scrivere la stessa cosa.
                                # N.B.: Ricorda anche che riguardo alle prezzo_tutte_nuove_A_in_quantita_EI che il costo totale di tutti gli investimenti di espansione necessari che sono in numero di macchine EI ci troviamo dentro all'if dell'ipotesi differenza_debito>costo_tutte_EI per cui sicuramente si possono comprare tutte le EI che servono col debito disponibile.
                                # Invece per quanto riguarda alle prezzo_tutte_replacement_macchine_A_RI è il costo totale della quantità di macchine di replacement che puoi permetterti col debito disponibile rimasto, per cui non sono tutto il numero RI che sarebbe necessario
                                # ma solo quelle che _j può permettersi e quindi anche queste sei sicuro che il loro costo stia dentro al debito disponibile
                                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = 0 # deve consumare tutto il suo NW (il cash) siccome non è sufficiente a coprire le spese e poi si indebita per il restante
                            else:
                                # ovvero il caso che: NW - c_jQ_j - costo_tot_EI - costo_tot_RI > 0
                                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzo_tutte_replacement_macchine_A_RI - prezzo_tutte_nuove_A_in_quantita_EI - costo_totale_della_produzione_c_j_t_Q_j_t
                                debito_totale_j_con_nuovo_ora = Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] # questa era in realtà come era stato già inizializzato sopra la prima volta, quindi si potrebbe anche cancellare questa riga, è superflua.
                                # poichè non c'è stato bisogno di ricorrere al debito siccome _j aveva abbastanza cash per fare tutto quello che voleva, MA IL RAPPORTO DEBITO-VENDITE < LAMBDA. INVECE PIU' AVANTI VEDREMO IL CASO DOVE NW_j BASTANO A COPRIRE TUTTE LE SPESE MA IL RAPPORTO E' GIA' MAGGIORE O UGUALE A LAMBDA
                            Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] = debito_totale_j_con_nuovo_ora # aggiorno, per il prossimo turno
                            # questo si poteva anche mettere fuori dall'if e aggiornarlo una volta sola per tuti i casi invece di ripeterlo ad ogni if - vedi i seguenti - però così facendo sei sicuro di non commettere errori con le rientranze
                        else:
                            # E' l'else riferito a "if differenza_debito > prezzo_tutte_nuove_A_in_quantita_EI:"
                            RI_j_t_replacement_investimenti = 0
                            prezzo_tutte_replacement_macchine_A_RI = 0
                            EI_j_t_espansione_investimenti = 0 # poi saranno aggiornate
                            prezzo_tutte_nuove_A_in_quantita_EI = 0 # poi aggiornate
                            while True:
                                if differenza_debito >= (prezzo_tutte_nuove_A_in_quantita_EI + prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure]):
                                    # ne aggiungo uno alla volta
                                    EI_j_t_espansione_investimenti = EI_j_t_espansione_investimenti + 1
                                    prezzo_tutte_nuove_A_in_quantita_EI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] + prezzo_tutte_nuove_A_in_quantita_EI
                                else:
                                    break
                            # Oppure si poteva anche fare così senza il while sopra più semplicemente, ma meno sicuro:
                            # EI_j_t_espansione_investimenti = int(differenza_debito / prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure])
                            debito_totale_j_con_nuovo_ora = (prezzo_tutte_nuove_A_in_quantita_EI + costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]) + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
                            NW_j_t_liquid_assets_consumption[azienda_j_consumption] = 0
                            numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure] = EI_j_t_espansione_investimenti + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure]
                            quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = EI_j_t_espansione_investimenti + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                    else:
                        # E' l'else riferito a "if len(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]) != 0:"
                        # se invece non hanno neanche una azienda capital da cui ordinare poichè non hanno ricevuto nemmeno una brochure
                        EI_j_t_espansione_investimenti = 0
                        RI_j_t_replacement_investimenti = 0
                        prezzo_tutte_nuove_A_in_quantita_EI = 0
                        prezzo_tutte_replacement_macchine_A_RI = 0
                        EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption] = 0
                        # Ora c'è parte simile a quanto già fatto sopra per un altro "else".
                        # Infatti non sappiamo se NW_j è stato sufficiente ad assorbire tutti i costi di produzione e se è rimasto cash (cioè NW>cQ) o no (esattamente NW=cQ)
                        # oppure non è stato sufficiente (siccome NW<cQ) e quanto debito è stato consumato.
                        if (NW_j_t_liquid_assets_consumption[azienda_j_consumption] - costo_totale_della_produzione_c_j_t_Q_j_t) <= 0:
                            debito_totale_j_con_nuovo_ora = (costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]) + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
                            NW_j_t_liquid_assets_consumption[azienda_j_consumption] = 0 # deve consumare tutto il suo NW (il cash) siccome non è sufficiente a coprire le spese e poi si indebita per il restante
                        else:
                            # ovvero il caso che: NW - c_jQ_j > 0
                            NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - costo_totale_della_produzione_c_j_t_Q_j_t
                            debito_totale_j_con_nuovo_ora = Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] # questa era in realtà come era stato già inizializzato sopra la prima volta, quindi si potrebbe anche cancellare questa riga, è superflua.
                            # poichè non c'è stato bisogno di ricorrere al debito siccome _j aveva abbastanza cash per fare tutto quello che voleva.
                        Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] = debito_totale_j_con_nuovo_ora # aggiorno, per il prossimo turno
                else:
                    # E' l'else riferito a "if differenza_debito > 0:" sopra. Ovvero il caso che  differenza_debito == 0. Invece non è possibile che sia <0 poichè è escluso siccome dovrebbe essere (c_j_t Q_j_t - NW_j_t)>quanto_debito_ancora che invece, se siamo arrivati fino a qui, era "False"
                    # Uno potrebbe pensare che se siamo qua ad eseguire allora vuol dire che "differenza_debito <= 0" invece NO! Il caso "differenza_debito < 0" è fuori discussione.
                    # In realtà l'unico caso possibile se siamo qua è che "differenza_debito == 0", poichè il caso "differenza_debito= quanto_deb_ancora_fare - (cQ-NW) < 0" implica che (cQ-NW)>quanto_deb_ancora_fare che è invece escluso poichè ci troviamo nel suo "else":
                    # l'if che ci ha mandati qua si accede dall'else, per cui siamo nel caso "(cQ-NW) <= quanto_deb_ancora_fare" ma siccome "differenza_debito==0" allora sicuramente "(cQ-NW) == quanto_deb_ancora_fare" !!!
                    # Il debito disponibile da consumare basta a pagare esattamente tutta e sola la produzione completa che NW_j non basta a pagare: ovviamente è un caso molto molto raro che si finisca in questa parte del programma.
                    debito_totale_j_con_nuovo_ora = quanto_debito_si_puo_ancora_fare + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] # Siccome qua vale quanto_debito_si_puo_ancora_fare = (costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]):
                    # debito_totale_j_con_nuovo_ora = (costo_totale_della_produzione_c_j_t_Q_j_t - NW_j_t_liquid_assets_consumption[azienda_j_consumption]) + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
                    # Infatti con l'altra scrittura sarebbe stato identico siccome:
                    # debito_totale_j_con_nuovo_ora = quanto_debito_si_puo_ancora_fare - (differenza_debito==0) + Deb_j_turno_precedente_debito_consumption[azienda_j_consumption]
                    NW_j_t_liquid_assets_consumption[azienda_j_consumption] = 0 # deve consumare tutto il suo il cash siccome (cQ-NW) == quanto_deb_ancora_fare" e NW < cQ
                    EI_j_t_espansione_investimenti = 0
                    RI_j_t_replacement_investimenti = 0 # pero' cosi' le macchine piu' vecchie di eta saranno buttate via e non sostituite e cosi' il # delle A in Theta_j calerà
                    prezzo_tutte_nuove_A_in_quantita_EI = 0
                    prezzo_tutte_replacement_macchine_A_RI = 0
                    Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] = debito_totale_j_con_nuovo_ora # aggiorno, per il prossimo turno
        else:
            # E' l'else riferito a "if rapporto_debito_vendite_j_turno_precedente < lambda_max_debt_sale_ratio:" di sopra.
            # Quindi _j è rationed, ma non vuol dire che non esista il caso che abbia abbastanza cash NW_j per finanziare non solo tutta la produzione c_jQ_j ma anche l'espansione EI e i replacement RI!
            # BISOGNA RIPETERE MODIFICANDOLO TUTTO QUANTO SCRITTO SOPRA, COMPRESI CALCOLO EI e RI, POICHE' IN QUESTA PARTE DEL PROGRAMMA NON SONO MAI STATI ESEGUITI QUEI CALCOLI!
            # ----- NO DEBITO LAMBDA -----
            if costo_totale_della_produzione_c_j_t_Q_j_t > NW_j_t_liquid_assets_consumption[azienda_j_consumption]:
                # allora non può finanziare l'intera produzione poichè NW non basta, qua ne può finanziare al massimo NW_j e quindi quanti lavoratori e quantità potrà al massimo avere
                produzione_permessa = NW_j_t_liquid_assets_consumption[azienda_j_consumption]
                quantità_Q_permessa = produzione_permessa/c_j_t_unit_cost_production
                lavoratori_L_effettivamente_permessi_da_assumere = quantità_Q_permessa/pi_j_t_average_productivity
                # siccome in realtà i lavoratori da assumere sono meno di quelli che si sarebbe voluto ma che abbiamo già sottratto dai disoccupati totali, avendo tolto troppo, dobbiamo riaggiungere ai disoccupati quelli che alla fine non sono stati assunti
                persone_ancora_disoccupate = persone_ancora_disoccupate + int( np.round(L_D_j_t_labor_demand - lavoratori_L_effettivamente_permessi_da_assumere) ) # poichè  L_D_j_t_labor_demand > lavoratori_L_effettivamente_permessi_da_assumere
                occupati_nelle_consumption -= ( np.round(L_D_j_t_labor_demand - lavoratori_L_effettivamente_permessi_da_assumere) )
                L_D_j_t_labor_demand = int( np.round(lavoratori_L_effettivamente_permessi_da_assumere) ) # aggiorno con un bagno di realtà
                Q_j_t_nel_paper_non_dice_come = np.round(quantità_Q_permessa) # aggiorno con un bagno nella realtà
                costo_totale_della_produzione_c_j_t_Q_j_t = c_j_t_unit_cost_production * Q_j_t_nel_paper_non_dice_come # lo aggiorno poichè dopo se userò "costo_totale_della_produzione_c_j_t_Q_j_t" deve essere solo quella effettivamente fatta
                EI_j_t_espansione_investimenti = 0 # poichè non riesci nemmeno ad assumere tutte i lavoratori per produrre la quantità che volevi figuriamoci comprare le macchine
                RI_j_t_replacement_investimenti = 0 # pero' cosi' le macchine piu' vecchie di eta saranno buttate via e non sostituite e cosi' il # delle A in Theta_j calerà
                prezzo_tutte_nuove_A_in_quantita_EI = 0
                prezzo_tutte_replacement_macchine_A_RI = 0
                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - costo_totale_della_produzione_c_j_t_Q_j_t # devi consumare tutto il tuo NW (il cash), prima di indebitarti per il restante però siccome non puoi usare quantità non intere potrebbe essere < NW 
            else:
                # quindi siamo nel caso che costo_totale_della_produzione_c_j_t_Q_j_t <= NW_j_t_liquid_assets_consumption[azienda_j_consumption]
                # Anche se _j è rationed e non può indebitarsi, il cash NW_j è sufficiente per tutta la produzione e ora vediamo per cosa altro...
                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - costo_totale_della_produzione_c_j_t_Q_j_t
                if NW_j_t_liquid_assets_consumption[azienda_j_consumption] > 0:
                    # siccome sono rimasti dei soldi NW_j ha senso guardare espansione e replacement.
                    # -----
                    # -----
                    # HO ACCORPATO QUA ANCHE LA PARTE PER CALCOLARE "EI_j_t_espansione_investimenti" CHE ERA, PRIMA, INIZIALMENTE ALL'INIZIO DEL LOOP PER LE _j, SICCOME NON HA SENSO CALCOLARE QUANTE NUOVE MACHINE EI VORREBBERO SE POI NON LE POSSONO COMPRARE POICHE' E' FINITO NW_j, SPRECO DI TEMPO!
                    # K_desid_j_t_capitale_auspicato = Q_desid_j_t/u_desired_level_capacity_utilization # come scrivono loro nel paper
                    K_desid_j_t_capitale_auspicato = Q_desid_j_t/(pi_j_t_average_productivity * u_desired_level_capacity_utilization) # mia supposizione
                    K_trig_j_soglia_per_investire = K_j_t_macchine_turno_precedente * (1 + alpha_trigger_rule)
                    # FORSE QUESTA PARTE NON ANDRA' TENUTA NELLA VERSIONE DEL 2010
                    if K_desid_j_t_capitale_auspicato < np.round(K_trig_j_soglia_per_investire): # sennò quando sono uguali non funziona perchè ci sono sempre degli altri numeri che mette Python dopo tanti zeri, es. tu ti immagini sia 880.0<880.0: False, invece in realtà è 880.0<880.0000001: True !!!
                        EI_j_t_espansione_investimenti = 0
                    else:
                        EI_j_t_espansione_investimenti = int( np.round(K_trig_j_soglia_per_investire - K_j_t_macchine_turno_precedente) ) # per cui espande sempre e solo di alfa*K. Così è nel 2006 e nel 2008, cambierà invece nel paper successivo cioè il 2010
                    # STRANO che non sia K_desid_j_t_capitale_auspicato invece di K_trig_j_soglia_per_investire ed effettivamente nel 2010, al contrario del 2006 e 2008, è proprio così!!! Ovvero:
                        # EI_j_t_espansione_investimenti = int( np.round(K_desid_j_t_capitale_auspicato - K_j_t_macchine_turno_precedente) )
                    # simile a quella presente sotto, ma diversa poichè è per EI e non per RI
                    if len(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]) != 0:
                        # poichè esiste il caso che alla azienda consumption _j non sia arrivata nemmeno una brochure da nessuna capital _i
                        valore_eq_da_minimizzare = np.array([]) # alla fine del "for" seguente, questo vettore avrà la stessa dimensione del np.array brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]. Infatti qua valuti l'eq. per tutti non solo per quelle brochure che hanno superato la prova con la A app. Theta_j(t), poichè quella A è da buttare perchè vecchia
                        for indice_azienda_capital_brochure in brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]:
                            # questa è l'equazione nel paper 2010. In realtà sarebbe:  p*_i + b c_i  ma c'è il salario w(t) al numeratore di entrambe le frazioni per cui si può cancellare
                            eq_paper_2010_migliore_il_piu_piccolo = (1/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) + b_payback_period*(1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure])
                            # Nel caso che invece usi quello che dice nel 2006 a pag.7: "highest productivity/price ratio" anche se lascio come nome il più piccolo per non dover cambiare tutto anche se ora bisogna prendere il massimo e non il minimo. Ciò che hai pensato è che il bug sia qua.
                            # Infatti come era prima preferivi b volte di più quelli che hanno un alto A rispetto a quelli che hanno un alto B quindi due che hanno lo stesso valore come minimo dovrebbero avere uno la B b volte minore della A dell'altro.
                            # Per quello viene che la A media è sempre superiore della B media nel grafico poichè sistematicamente quelle che hanno estratto alto B e basso A muoiono poichè vengono preferiti quelli che hanno estratto un alto A e un basso B ben b volte di più!
                            #eq_paper_2010_migliore_il_piu_piccolo = aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]/prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[indice_azienda_capital_brochure]
                            valore_eq_da_minimizzare = np.append(valore_eq_da_minimizzare, eq_paper_2010_migliore_il_piu_piccolo)
                        # attento che se ce ne sono due o più con lo stesso minimo, fornisce la posizione nell'np.array (l'indice) solo del primo di essi
                        posizione_minimo = np.argmin(valore_eq_da_minimizzare)
                        #posizione_minimo = np.argmax(valore_eq_da_minimizzare) # se invece prendo il valore massimo poichè massimizzo il rapporto A/p ovvero B(A/w) ma w è lo stesso per tutte le _i quindi è una costante per cui AB/const diventa all'atto pratico che massimizzo la moltiplicazione AB
                        vincitore_indice_capital_azienda_nella_brochure = brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption][posizione_minimo]
                        EI_indici_capital_firms_con_cui_espandere_theta_in_consumption_firms[azienda_j_consumption] = vincitore_indice_capital_azienda_nella_brochure
                        # prezzo_di_una_sola_nuova_A_selezionata_per_EI = w_salario_questo_turno/aziende_capital_good_B_i_tau[vincitore_indice_capital_azienda_nella_brochure]
                        # OPPURE SE IL PREZZO DEI MACCHINARI A VIENE FISSATO CON ANCHE IL MARK-UP COME NEL 2010 mi_1=0.04 ALLORA SI DEVE USARE IL VETTORE PREZZI POICHE' NEL TURNO PRECEDENTE AVRAI MESSO TUTTI I PREZZI COMPRESI DI mi_1 LI'
                        prezzo_di_una_sola_nuova_A_selezionata_per_EI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] # nel caso 2010 e che ci sia parametro_scala_roventini per le capital
                        #prezzo_tutte_nuove_A_in_quantita_EI = int( np.round(EI_j_t_espansione_investimenti) ) * prezzo_di_una_sola_nuova_A_selezionata_per_EI
                        prezzo_tutte_nuove_A_in_quantita_EI = np.round(EI_j_t_espansione_investimenti) * prezzo_di_una_sola_nuova_A_selezionata_per_EI
                    # -----
                    # -----
                        if NW_j_t_liquid_assets_consumption[azienda_j_consumption] > prezzo_tutte_nuove_A_in_quantita_EI:
                            NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzo_tutte_nuove_A_in_quantita_EI # tolgo la cifra pagata per gli investimenti di espansione EI
                            numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure] = int( np.round(EI_j_t_espansione_investimenti) ) + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure]
                            quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = int( np.round(EI_j_t_espansione_investimenti) ) + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                            # -----
                            # -----
                            # -----
                            RI_j_t_replacement_investimenti = 0 # replacement investment poi verrà aggiornato
                            prezzo_tutte_replacement_macchine_A_RI = 0 # poi verrà aggiornato
                            while True:
                                for indice_A_posseduta_appartenente_theta_j_t in range(K_j_t_macchine_turno_precedente):
                                    A_posseduta_appartenente_theta_j_t = aziende_consumption_good[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t]
                                    #if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= eta_max_machine_age:
                                    if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= eta_max_machine_age-2:
                                        # Il vincitore lo conosci già dal calcolo fatto prima per EI che è lo stesso calcolo. Nota che il calcolo sopra viene fatto anche nel caso in cui il calcolo all'inizio del programma desse EI_j_t_espansione_investimenti=0 per cui in realtà non ci sarebbe bisogno di cercare macchine con cui espandere la produzione.
                                        # E' meglio fatto così poichè sopra la ricerca del vincitore viene fatta una volta sola per ogni consumption _i, invece qua sarebbe stata fatta per ogni A con età>=eta quindi ripetevi un sacco di volte quel calcolo per trovare sempre lo stesso vincitore, sarebbe stupido!!!
                                        # La differenza qua rispetto alla parte sopra per EI, sta nelle prossime 2 righe:
                                        if NW_j_t_liquid_assets_consumption[azienda_j_consumption] >= prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure]:
                                            RI_j_t_replacement_investimenti = RI_j_t_replacement_investimenti + 1 # aggiorno poichè la macchina è da cambiare perchè troppo vecchia e l'azienda j se la può permettere con i suoi fondi NW_j.
                                            NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] # sottraggo la spesa appena fatta
                                            prezzo_tutte_replacement_macchine_A_RI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] + prezzo_tutte_replacement_macchine_A_RI
                                            RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption] = np.append(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption], indice_A_posseduta_appartenente_theta_j_t)
                                            indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption] = np.append(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption], vincitore_indice_capital_azienda_nella_brochure)
                                            numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure] = 1 + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure]
                                            quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = 1 + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                                            RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = 1 + RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                                        else:
                                            break
                                    else:
                                        # Devi farlo così, con i loop di for poichè, anche se sembra una buona idea, non può funzionare fare i conti operando sui vettori in un solo colpo invece che su numeri singoli: questo a causa del fatto che stai usando indici che rimandano a posizioni diverse in quei vettori!
                                        indici_candidati_prova_payback_superata = np.array([], int)
                                        valore_rapporto_candidati_superata = np.array([])
                                        for indice_azienda_capital_brochure in brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]:
                                            if (1/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):
                                            # Invece questo è il caso in cui i prezzi dei macchinari sono parametro*salario/B poichè servono parametro lavoratori per fare B macchinari:
                                            # if (scala_lavoratori_capital_good_parametro_mail_roventini/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):# investimenti non periodici perchè anche RI 2 turni consecutivi stesso macchinario
                                            # Questo è il caso in cui non si può sostituire una A prima di b turni sennò una RI potrebbe essere sostituita ogni turno e in cui i prezzi dei macchinari sono parametro*salario/B poichè servono parametro lavoratori per fare B macchinari:
                                            # if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= b_payback_period and (scala_lavoratori_capital_good_parametro_mail_roventini/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):# variabile
                                            # if eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][indice_A_posseduta_appartenente_theta_j_t] >= b_payback_period and (parametro_mail_roventini_scala_lavoratori/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure]) <= b_payback_period * ( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ):# fisso
                                                indici_candidati_prova_payback_superata = np.append(indici_candidati_prova_payback_superata, indice_azienda_capital_brochure)
                                                minimizzare_il_migliore_il_piu_piccolo = abs( (1/aziende_capital_good_B_i_tau[indice_azienda_capital_brochure])/( (1/A_posseduta_appartenente_theta_j_t) - (1/aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]) ) )
                                                # Nel caso che invece usi quello che dice nel 2006 a pag.7: "highest productivity/price ratio" anche se lascio come nome il più piccolo per non dover cambiare tutto anche se ora bisogna prendere il massimo e non il minimo.
                                                #minimizzare_il_migliore_il_piu_piccolo = aziende_capital_good_A_i_tau[indice_azienda_capital_brochure]/prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[indice_azienda_capital_brochure]
                                                valore_rapporto_candidati_superata = np.append(valore_rapporto_candidati_superata, minimizzare_il_migliore_il_piu_piccolo)
                                        if len(indici_candidati_prova_payback_superata) > 0:
                                            # ovvero se qualcuno ha vinto: infatti la A posseduta in Theta_j(t) potrebbe essere migliore di tutte le A delle brochure
                                            # attento che se ce ne sono due o più con lo stesso minimo, fornisce la posizione nell'np.array (l'indice) del primo di essi
                                            posizione_minimo = np.argmin(valore_rapporto_candidati_superata)
                                            #posizione_minimo = np.argmax(valore_rapporto_candidati_superata) # se invece prendo il valore massimo poichè massimizzo il rapporto A/p ovvero B(A/w) ma w è lo stesso per tutte le _i quindi è una costante per cui AB/const diventa all'atto pratico che massimizzo la moltiplicazione AB
                                            vincitore_indice_capital_azienda = indici_candidati_prova_payback_superata[posizione_minimo]
                                            if NW_j_t_liquid_assets_consumption[azienda_j_consumption] >= prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda]:
                                                RI_j_t_replacement_investimenti = RI_j_t_replacement_investimenti + 1 # aggiorno poichè la macchina è da cambiare e l'azienda j se la può permettere pagando di tasca sua con NW_j.
                                                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda] # sottraggo la spesa appena fatta
                                                prezzo_tutte_replacement_macchine_A_RI = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda] + prezzo_tutte_replacement_macchine_A_RI
                                                RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption] = np.append(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption], indice_A_posseduta_appartenente_theta_j_t)
                                                indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption] = np.append(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption], vincitore_indice_capital_azienda)
                                                numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda] = 1 + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda]
                                                quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption] = 1 + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption]
                                                RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption] = 1 + RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda, azienda_j_consumption]
                                            else:
                                                break
                            # -----
                            # -----
                            # -----
                                # SICCOME DEVE ESSERCI UNA INTERRUZIONE AL WHILE INFINITO poichè potrebbe essere che se le A di Theta_j sono giovani e sono le migliori non vengano mai incontrati i due break messi sopra!
                                # SECONDO ME DEVE STARE SULLA STESSA RIENTRANZA DEL "for indice_A_posseduta_appartenente_theta_j_t in range(K_j_t_macchine_turno_precedente):" POICHE' QUANDO è FINITO E SONO SCORSE TUTTE LE A di Theta_j DEVE INTERROMPERE IL WHILE!
                                break
                            # Qua in poi riprende e ci troviamo sulle stesse righe (ovvero rientranza) di quelle sotto (cioè che seguono) "if NW_j_t_liquid_assets_consumption[azienda_j_consumption] > prezzo_tutte_nuove_A_in_quantita_EI:"
                        else:
                            # E' l'else riferito a NO DEBITO LAMBDA "if NW_j_t_liquid_assets_consumption[azienda_j_consumption] > prezzo_tutte_nuove_A_in_quantita_EI:" sopra.
                            EI_j_t_espansione_investimenti = 0
                            while True:
                                if NW_j_t_liquid_assets_consumption[azienda_j_consumption] >= prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure]:
                                    EI_j_t_espansione_investimenti = EI_j_t_espansione_investimenti + 1
                                    NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure] # sottraggo la spesa appena fatta
                                else:
                                    break
                            # Oppure si poteva anche fare così senza il while sopra più semplicemente ma meno sicuro:
                            # EI_j_t_espansione_investimenti = int(NW_j_t_liquid_assets_consumption[azienda_j_consumption] / prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure])
                            # NW_j_t_liquid_assets_consumption[azienda_j_consumption] = NW_j_t_liquid_assets_consumption[azienda_j_consumption] - prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure]*int(NW_j_t_liquid_assets_consumption[azienda_j_consumption]/prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure])
                            prezzo_tutte_nuove_A_in_quantita_EI = EI_j_t_espansione_investimenti * prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[vincitore_indice_capital_azienda_nella_brochure]
                            numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure] = EI_j_t_espansione_investimenti + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[vincitore_indice_capital_azienda_nella_brochure]
                            quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption] = EI_j_t_espansione_investimenti + quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[vincitore_indice_capital_azienda_nella_brochure, azienda_j_consumption]
                            RI_j_t_replacement_investimenti = 0
                            prezzo_tutte_replacement_macchine_A_RI = 0
                    else:
                        # E' l'else riferito a NO DEBITO LAMBDA "if len(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]) != 0:" sopra.
                        # Siccome non c'è nessuna azienda _i da cui ordinare
                        EI_j_t_espansione_investimenti = 0
                        RI_j_t_replacement_investimenti = 0
                        prezzo_tutte_nuove_A_in_quantita_EI = 0
                        prezzo_tutte_replacement_macchine_A_RI = 0
                        EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption] = 0
                else:
                    # E' l'else riferito a NO DEBITO LAMBDA "if NW_j_t_liquid_assets_consumption[azienda_j_consumption] > 0:" sopra. Ovvero il caso che  NW_j_t_liquid_assets_consumption[azienda_j_consumption]==0.
                    # Invece non è possibile che sia <0 poichè è escluso siccome dovrebbe essere costo_totale_della_produzione_c_j_t_Q_j_t > NW_j_t_liquid_assets_consumption[azienda_j_consumption] che invece, se siamo arrivati fino a qui, era "False"
                    EI_j_t_espansione_investimenti = 0
                    RI_j_t_replacement_investimenti = 0
                    prezzo_tutte_nuove_A_in_quantita_EI = 0
                    prezzo_tutte_replacement_macchine_A_RI = 0
                    EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption] = 0
            # ----- NO DEBITO LAMBDA -----
        # Qua sono fuori dal loop da cui è iniziato tutto: "if rapporto_debito_vendite_j_turno_precedente < lambda_max_debt_sale_ratio:".            
        Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] = debito_totale_j_con_nuovo_ora # aggiorno, per il prossimo turno
        EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption] = EI_j_t_espansione_investimenti
        if azienda_j_consumption == consumption_da_seguire_e_stampare:#2
            print('----------------------------------------------------')
            print('La consumption', azienda_j_consumption,'nel turno', turno,'può fare debito per', quanto_debito_si_puo_ancora_fare, 'e il suo rapporto debito vendite è', rapporto_debito_vendite_j_turno_precedente)
            print('Gli NW_j_t_liquid_assets_consumption ad inizio turno prima di spenderli erano', soldi_liquidi_j_prima_di_spenderli,'e ora sono', NW_j_t_liquid_assets_consumption[azienda_j_consumption],'. I magazzini hanno merce rimanente N_j per', N_j_turno_precedente_merce_invenduta[azienda_j_consumption],'e vuole produrre in quantità', Q_j_t_nel_paper_non_dice_come)
            print('Il costo_totale_della_produzione_c_j_t_Q_j_t è', costo_totale_della_produzione_c_j_t_Q_j_t,', la Q_j_t_max_possibile_con_Theta_attuale è',Q_j_t_max_possibile_con_Theta_attuale,'e il numero di macchine K che possiede è',K_j_t_macchine_turno_precedente)
            print('Ha EI_j_t_espansione_ di', EI_j_t_espansione_investimenti, 'e RI_j_t_replacement_ di', RI_j_t_replacement_investimenti, '. Il prezzo_tutte_nuove_A_in_quantita_EI è', prezzo_tutte_nuove_A_in_quantita_EI, 'e il prezzo_tutte_replacement_macchine_A_RI è', prezzo_tutte_replacement_macchine_A_RI, '.')
            print('La domanda attesa D_expect_j_t è', D_expect_j_t,', Q_desid_j_t è', Q_desid_j_t,', N_desid_j_t_scorte_magazzino è', N_desid_j_t_scorte_magazzino,'e il K_desid_j_t_capitale_auspicato è', K_desid_j_t_capitale_auspicato)
            # print('Infine EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare è', EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption])
        Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption] = Q_j_t_nel_paper_non_dice_come
        Q_j_quantita_prodotta_turno_precedente[azienda_j_consumption] = Q_j_t_nel_paper_non_dice_come
        # I_j(t)= EI_j(t) + RI_j(t) verrà eseguito nel loop delle capital _i  invece che qua, in quello delle consumption, poichè è più facile fare i conti; tanto gli investimenti delle _j costituiscono il reddito delle _i
    D_expect_j_o_domanda_turno_precedente_storico_tutti_turni = np.append(D_expect_j_o_domanda_turno_precedente_storico_tutti_turni, np.sum(D_j_domanda_turno_precedente) )    
        
        
    # Se vuoi invece che seguano un ordine casuale così se c'è scarsità di lavoratori non accade che l'ultima azienda rimane sistematicamente fregata ad ogni turno poichè essendo sempre l'ultima della coda non può assumere siccome tutta la popolazione ha già un lavoro!
    # lista_capital_da_rimescolare = [v for v in range(F2_numero_capital_industry)]
    # np.random.shuffle(lista_capital_da_rimescolare) # attento che non crea una nuova lista ma scombussola quella originale!
    # for azienda_i_capital in lista_capital_da_rimescolare:
    for azienda_i_capital in range(F2_numero_capital_industry):
        # Assume i lavoratori controllando che ce ne siano abbastanza rimasti disoccupati e in base a quanti riesce a trovarne produce quel che può restituendo i soldi NW_j alla j se non riesce a produrre tutte le maccchine che doveva
        Q_i_t_quante_A_macchine_deve_produrre = np.sum(quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :])
        #L_D_i_t_labor_demand = int( np.round(Q_i_t_quante_A_macchine_deve_produrre/aziende_capital_good_B_i_tau[azienda_i_capital]) )
        L_D_i_t_labor_demand = int( np.round( scala_lavoratori_capital_good_parametro_mail_roventini*(Q_i_t_quante_A_macchine_deve_produrre/aziende_capital_good_B_i_tau[azienda_i_capital]) ) )
        # if produttivita_economia_che_uso_in_scala_lav_rov < aziende_capital_good_B_i_tau[azienda_i_capital]:
            # per produrre 1 macchinario A serve un'unità minima di lavoro.
            # unita_minima_lavoro_per_azienda_i = parametro_mail_roventini_scala_lavoratori
        # else:
            # credo infatti che l'unità minima di lavoro sia rov(t)/B_i(t). Dove rov(t)=rov_param*A_media(t). Ed essa produce A_media(t)/B_i(t). E' simile come era prima e al turno=1 dove essendo B_i(iniziale)=1 ogni scala_lav impiegati da _i producevano 1 macchinario A_i
            # unita_minima_lavoro_per_azienda_i = np.round( (produttivita_economia_che_uso_in_scala_lav_rov/aziende_capital_good_B_i_tau[azienda_i_capital]) * parametro_mail_roventini_scala_lavoratori )
        # L_D_i_t_labor_demand = unita_minima_lavoro_per_azienda_i * Q_i_t_quante_A_macchine_deve_produrre
        if Q_i_t_quante_A_macchine_deve_produrre < aziende_capital_good_B_i_tau[azienda_i_capital] and Q_i_t_quante_A_macchine_deve_produrre > 0:
        # L'azienda _i però dovrà pagare un salario più alto di quello che produce così e siccome incassa solo quello che produce potrebbe non rientrare nemmeno dei costi di produzione c_i e ritrovarsi con NW_i<0 fallendo!
            #L_D_i_t_labor_demand = L_D_i_t_labor_demand + 1 # poichè int(0._ _) fa 0 quindi può essere che avanti nella simulazione la B_i>>Q_i e non viene assunto neanche un lavoratore!
            L_D_i_t_labor_demand = L_D_i_t_labor_demand + 1*scala_lavoratori_capital_good_parametro_mail_roventini # poichè int(0._ _) fa 0 quindi può essere che avanti nella simulazione la B_i>>Q_i e non viene assunto neanche un lavoratore!
        costo_totale_della_produzione_c_i_t_Q_i_t = c_i_t_unit_cost_production[azienda_i_capital] * Q_i_t_quante_A_macchine_deve_produrre
        if NW_i_t_liquid_assets_capital[azienda_i_capital] < costo_totale_della_produzione_c_i_t_Q_i_t:
            Q_i_t_quante_A_macchine_deve_produrre = int(NW_i_t_liquid_assets_capital[azienda_i_capital] / c_i_t_unit_cost_production[azienda_i_capital])
            # L_D_i_t_labor_demand = int( Q_i_t_quante_A_macchine_deve_produrre/aziende_capital_good_B_i_tau[azienda_i_capital] )
            L_D_i_t_labor_demand = int( scala_lavoratori_capital_good_parametro_mail_roventini*(Q_i_t_quante_A_macchine_deve_produrre/aziende_capital_good_B_i_tau[azienda_i_capital]) )
            # L_D_i_t_labor_demand = unita_minima_lavoro_per_azienda_i * Q_i_t_quante_A_macchine_deve_produrre
        if persone_ancora_disoccupate >= L_D_i_t_labor_demand:
            persone_ancora_disoccupate = persone_ancora_disoccupate - L_D_i_t_labor_demand
            occupati_nelle_capital += L_D_i_t_labor_demand
        else:
            if persone_ancora_disoccupate < 1*scala_lavoratori_capital_good_parametro_mail_roventini:
            # if persone_ancora_disoccupate < unita_minima_lavoro_per_azienda_i:
                # Se le _j assumono tutti i lavoratori poi le _i non possono produrre macchine anche se ne avrebbero la possibilità e falliscono: non è giusto perchè le _j essendo fatte prima hanno la precedenza nelle assunzioni!
                # Allora mi sono inventato questo sistema, per cui a caso una _j perde lavoratori sufficienti a produrre una sola macchina (o più a piacimento) così comunque vada le _i che avevano ordini non falliscono a causa della piena occupazione
                array_strano_j_producono = np.where(Q_j_t_quantita_prodotta_nel_turno_attuale > 0) # funziona anche se non sono interi 0 ma 0.0
                array_con_indici_delle_j_che_hanno_prodotto = np.copy(array_strano_j_producono[0])
                if len(array_con_indici_delle_j_che_hanno_prodotto) > 0:
                    np.random.shuffle(array_con_indici_delle_j_che_hanno_prodotto)
                    consumption_estratta_perde_produzione_lavoratori_assunti = array_con_indici_delle_j_che_hanno_prodotto[0]
                    persone_ancora_disoccupate = 1*scala_lavoratori_capital_good_parametro_mail_roventini # libero i lavoratori che erano assunti affinchè siano disponibili per la _i
                    occupati_nelle_consumption -= 1*scala_lavoratori_capital_good_parametro_mail_roventini
                    # persone_ancora_disoccupate = unita_minima_lavoro_per_azienda_i
                    # occupati_nelle_consumption -= unita_minima_lavoro_per_azienda_i
                    if Q_j_t_quantita_prodotta_nel_turno_attuale[consumption_estratta_perde_produzione_lavoratori_assunti] - np.round( 1*scala_lavoratori_capital_good_parametro_mail_roventini * (np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) ) < 0:
                    # if Q_j_t_quantita_prodotta_nel_turno_attuale[consumption_estratta_perde_produzione_lavoratori_assunti] - np.round( unita_minima_lavoro_per_azienda_i * (np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) ) < 0:
                        NW_j_t_liquid_assets_consumption[consumption_estratta_perde_produzione_lavoratori_assunti] += w_salario_questo_turno * Q_j_t_quantita_prodotta_nel_turno_attuale[consumption_estratta_perde_produzione_lavoratori_assunti]/(np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) # ovvero i lavoratori effettivamente assunti sennò possono venire consumi negativi aggregati
                        Q_j_quantita_prodotta_turno_precedente[consumption_estratta_perde_produzione_lavoratori_assunti] = 0
                        Q_j_t_quantita_prodotta_nel_turno_attuale[consumption_estratta_perde_produzione_lavoratori_assunti] = 0
                    else:
                        Q_j_quantita_prodotta_turno_precedente[consumption_estratta_perde_produzione_lavoratori_assunti] -= np.round( 1 * scala_lavoratori_capital_good_parametro_mail_roventini * (np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) )
                        Q_j_t_quantita_prodotta_nel_turno_attuale[consumption_estratta_perde_produzione_lavoratori_assunti] -= np.round( 1 * scala_lavoratori_capital_good_parametro_mail_roventini * (np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) )
                        NW_j_t_liquid_assets_consumption[consumption_estratta_perde_produzione_lavoratori_assunti] += w_salario_questo_turno * 1 * scala_lavoratori_capital_good_parametro_mail_roventini # vengono restituiti i soldi pagati per il salario dei lavoratori assunti dalla _j
                        # Q_j_quantita_prodotta_turno_precedente[consumption_estratta_perde_produzione_lavoratori_assunti] -= np.round( unita_minima_lavoro_per_azienda_i * (np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) )
                        # Q_j_t_quantita_prodotta_nel_turno_attuale[consumption_estratta_perde_produzione_lavoratori_assunti] -= np.round( unita_minima_lavoro_per_azienda_i * (np.sum(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])/len(aziende_consumption_good[consumption_estratta_perde_produzione_lavoratori_assunti])) )
                        # NW_j_t_liquid_assets_consumption[consumption_estratta_perde_produzione_lavoratori_assunti] += w_salario_questo_turno * unita_minima_lavoro_per_azienda_i                        
            L_D_i_t_labor_demand = persone_ancora_disoccupate
            occupati_nelle_capital += persone_ancora_disoccupate
            persone_ancora_disoccupate = 0
            #Q_i_t_quante_A_macchine_deve_produrre = int( L_D_i_t_labor_demand * aziende_capital_good_B_i_tau[azienda_i_capital] )
            # Q_i_t_quante_A_macchine_deve_produrre = int( (L_D_i_t_labor_demand/scala_lavoratori_capital_good_parametro_mail_roventini) * aziende_capital_good_B_i_tau[azienda_i_capital] )
            potrebbero_essere_ai_turni_alti_molte_di_piu_di_Q_i_t_quante_A_macchine_deve_produrre = np.round( (L_D_i_t_labor_demand/scala_lavoratori_capital_good_parametro_mail_roventini) * aziende_capital_good_B_i_tau[azienda_i_capital] )
            # potrebbero_essere_ai_turni_alti_molte_di_piu_di_Q_i_t_quante_A_macchine_deve_produrre = np.round(L_D_i_t_labor_demand/unita_minima_lavoro_per_azienda_i)
            if Q_i_t_quante_A_macchine_deve_produrre > potrebbero_essere_ai_turni_alti_molte_di_piu_di_Q_i_t_quante_A_macchine_deve_produrre:
                Q_i_t_quante_A_macchine_deve_produrre = potrebbero_essere_ai_turni_alti_molte_di_piu_di_Q_i_t_quante_A_macchine_deve_produrre
        if Q_i_t_quante_A_macchine_deve_produrre < np.sum(quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :]):
            # Così prendi entrambi i casi, cioè che non si possa produrre tutti gli ordini o per mancanza di soldi per pagare i lavoratori o proprio per mancanza di essi, altrimenti avresti dovuto ripetere due volte il seguente codice uguale in entrambi gli if.
            # ora però questa quantità, inferiore a quella voluta, deve essere ripartita tra le aziende e bisogna decidere quali aziende _j e di quante macchine rimarranno senza
            array_strano_non_funziona = np.where(quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :] > 0) # funziona anche se non sono interi 0 ma 0.0
            # questo è un array strano poichè è fatto così ad esempio: (array([1, 3], dtype=int64),)
            # e se ne prendi il primo con nomearray[0] ottieni array([1, 3], dtype=int64)
            # vedi sito: https://stackoverflow.com/questions/33747908/output-of-numpy-wherecondition-is-not-an-array-but-a-tuple-of-arrays-why
            array_con_indici_delle_j_che_hanno_ordinato = np.copy(array_strano_non_funziona[0])
            # oppure direttamente:
            #array_con_indici_delle_j_che_hanno_ordinato = np.where(quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :] > 0)[0]
            # oppure anche ma dovrebbe essere uguale:
            #for ci in array_strano_non_funziona[0]:
                #array_con_indici_delle_j_che_hanno_ordinato = np.append(array_con_indici_delle_j_che_hanno_ordinato, ci)
            # queste sotto sono le macchine che NON è possibile produrre, è un vettore riga 1xF1 :
            array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte = np.copy(quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :])
            lista_estrazione_indici_consumption = []
            for indice_consumption_j_che_ha_ordinato in array_con_indici_delle_j_che_hanno_ordinato:
                quante_ne_ha_ordinate = quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, indice_consumption_j_che_ha_ordinato] # sicuramente è un numero =1 o >1
                for duplicato in range(quante_ne_ha_ordinate):
                    lista_estrazione_indici_consumption.append(indice_consumption_j_che_ha_ordinato)
            np.random.shuffle(lista_estrazione_indici_consumption) # non crea una nuova lista ma permuta casualmente i membri della lista
            #for posizione in range(Q_i_t_quante_A_macchine_deve_produrre):#fornisce errore quando Q_i_t_quante_A_macchine è un float invece che un int
            for posizione in range(np.int64(Q_i_t_quante_A_macchine_deve_produrre)):
                # siccome Q_i_t_quante_A_macchine_deve_produrre < len(lista_estrazione_indici_consumption)
                array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte[ lista_estrazione_indici_consumption[posizione] ] -= 1 # sottraggo la macchina che si riuscirà a produrre, ovvero quella della _j estratta
            # numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[azienda_i_capital] = numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[azienda_i_capital] - np.sum(array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte) # se vuoi farlo in un colpo solo
            array_macchine_manca_lavoro_solo_possibile_effettivamente_produrre = quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :] - array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte
            # in pratica sarebbe: macchine che si vorrebbe produrre - macchine che non si può produrre = macchine che effettivamente produrremo cioè NUMERO DI VOLTE CHE ABBIAMO ESTRATTO UNA CERTA _j
            # gli elementi di "array_copia_riga.." o sono = o sono < di quelli della riga della matrice
            # np.array (vettore riga diciamo) con F1 elementi (sono le _j): è un vettore 1xF1
            for posizione_risultato in range(len(array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte)):
                # ovvero range(F1_numero_consumption_industry) cioè le consumption _j
                risultato_sottrazione = array_macchine_manca_lavoro_solo_possibile_effettivamente_produrre[posizione_risultato]
                if array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte[posizione_risultato] != 0:
                    # infatti dove è 0 o non venivano prodotte del tutto macchine o sono state pescate tutte e quindi saranno prodotte quante se ne volevano, quindi non c'è nulla di strano,
                    # per quella _j non bisogna modificare nulla. In questo caso la differenza tra la quantità totale e quelle estratte è nulla risultato_sottrazione=0 ma non ci interessa poichè non viene preso dal =!0.
                    # Come dividi quelle che si possono produrre per la _j tra le sue richieste di RI_j e EI_j? Decido di dare la priorità alle EI_j
                    # il problema è che tu non sai (non hai pronto il dato) EI riferito ad ogni capital ma solo EI riferito ad ogni _j ma non sai da quali _i comprerà. Per cui devi calcolarlo
                    EI_j_di_questa_i = quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato] - RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato]
                    RI_j_di_questa_i = RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato]
                    RI_j_di_questa_i_promemoria = RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato] # perchè ti servirà questo dato (per il loop sotto) dopo che avrai già cambiato il valore sia della matrice che di RI_j qua sopra
                    if risultato_sottrazione >= EI_j_di_questa_i:
                        # EI è ok (potrebbe anche essere EI=0 dalla condizione sopra e sarebbe vero) ma non RI.
                        # Nel caso quella _j avesse ordinato dalla _i solo macchine per RI, nessuna per EI e nessuna è stata estratta, o almeno una di queste non è stata estratta; in questa parte staresti quindi cancellando gli indici delle A per RI.
                        # Non devi invece cancellare l'indice per gli EI della _i dalla _j con un comando del tipo: EI_indici_capital_firms_con_cui_espandere_theta_in_consumption_firms[posizione_risultato] = F2_numero_capital_industry + 2  poichè quella _j potrebbe invece aver ordinato A come EI da un'altra _i anche se non l'ha fatto da questa _i qua, credo !!!
                        RI_j_di_questa_i = risultato_sottrazione - EI_j_di_questa_i # quante RI_j_di_questa_i posso effettivamente produrre, invece RI_j_di_questa_i_promemoria è quante ne avrei volute produrre
                        RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato] = RI_j_di_questa_i
                        quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato] = risultato_sottrazione
                        numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[azienda_i_capital] = numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[azienda_i_capital] - array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte[posizione_risultato] # siccome queste sono totali cioè le A prodotte da _i per tutte le _j
                        contatore_cancellare_RS = 0 # servirà perchè ne devi cancellare
                        copia_array_RS_indici_capital_stampo_per_ricerca = np.copy(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato])
                        # siccome devi cancellare l'indice della _i per tutte le volte che non riesce a produrre le RI (e poi dovrai cancellare la A da sostituire dal vettore analogo/specchio poichè quella A deve essere tenuta) e quindi la dimensione dell'np.array cala mentre tu però sei ancora nel loop per cui è tutto scombinato
                        # quindi hai bisogno di uno stampo su cui fare il loop e trovare l'indice della _i e poi lo cancelli dall'originale invece che dallo stampo!
                        for posizione_nell_array_della_lista_RS in range(len(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato])):
                            if contatore_cancellare_RS < (RI_j_di_questa_i_promemoria - RI_j_di_questa_i):
                                # ovvero quanti ne devi cancellare poichè sono quante macchine _i non può produrre per quella _j: ad esempio se è (3-2) ne devi cercare 1 da cancellare per cui quando il contatore_cancellare=1 devi smettere. In realtà è una specie di While, sicuramente si poteva fare anche con While
                                if copia_array_RS_indici_capital_stampo_per_ricerca[posizione_nell_array_della_lista_RS] == azienda_i_capital:
                                    indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato] = np.delete(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato], posizione_nell_array_della_lista_RS - contatore_cancellare_RS)
                                    RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[posizione_risultato] = np.delete(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[posizione_risultato], posizione_nell_array_della_lista_RS - contatore_cancellare_RS)
                                    # serve togliere il contatore_cancellare_RS alla posizione nell'array poichè man mano che elimini una A l'np.array si rimpicciolisce per cui l'indice dell'array stampo non corrisponde più alla posizione della giusta A che intanto è slittata indietro di tante caselle quante sono le A già eliminate (che poi è il contatore_cancellare_RS)
                                    NW_j_t_liquid_assets_consumption[posizione_risultato] = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[azienda_i_capital] + NW_j_t_liquid_assets_consumption[posizione_risultato] # devi ridare i soldi a _j per la macchina pagata ma non prodotta da _i siccome è più comodo per la programmazione togliere a NW_j_t già i costi di tutte le macchine ordinate nel loop delle _j
                                    contatore_cancellare_RS = 1 + contatore_cancellare_RS
                    else:
                        # riferito a "if risultato_sottrazione >= EI_j_di_questa_i:"
                        # Si è deciso di dare priorità alle EI rispetto alle RI per la _j, come avviene per la loro scelta degli investimenti se non hanno NW_j e debito a sufficienza. Quindi _i non potendo produrre nemmeno tutti gli EI per _j abbiamo che
                        # RI è zero e EI=risultato_sottrazione ovvero tutti quelli possibili, avrai quindi un sacco di indici da cercare in RS ma ora anche nelle liste per EI (invece non è possibile il caso EI=0 dalla condizione sopra, poichè sarebbe anche RI=0)
                        EI_j_di_questa_i = risultato_sottrazione
                        RI_j_di_questa_i = 0
                        EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[posizione_risultato] = EI_j_di_questa_i # poichè le _j ordinano le EI (ma anche le RI per sostituire le macchine con età>=eta) tutte dalla STESSA _i che è la vincitrice tra le brochure con la b migliore!
                        RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato] = RI_j_di_questa_i
                        quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, posizione_risultato] = risultato_sottrazione
                        numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[azienda_i_capital] = numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption[azienda_i_capital] - array_macchine_non_possibile_produrre_riga_matrice_F2xF1_dove_sottratte_quelle_estratte[posizione_risultato]
                        contatore_cancellare_RS = 0 # servirà perchè ne devi cancellare, questa volta li devi eliminare tutti
                        copia_array_RS_indici_capital_stampo_per_ricerca = np.copy(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato])
                        for posizione_nell_array_della_lista_RS in range(len(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato])):
                            if contatore_cancellare_RS < RI_j_di_questa_i_promemoria:
                                # poichè questa volta, a differenza di prima, le devi cancellare tutte: ad esempio se ne devi cercare 3 da cancellare, per cui quando il contatore_cancellare=3 devi smettere poichè è partito da zero, dove ne ha già cancellata una.
                                if copia_array_RS_indici_capital_stampo_per_ricerca[posizione_nell_array_della_lista_RS] == azienda_i_capital:
                                    indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato] = np.delete(indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[posizione_risultato], posizione_nell_array_della_lista_RS - contatore_cancellare_RS)
                                    RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[posizione_risultato] = np.delete(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[posizione_risultato], posizione_nell_array_della_lista_RS - contatore_cancellare_RS)
                                    # serve togliere il contatore_cancellare_RS alla posizione nell'array poichè man mano che elimini una A l'np.array si rimpicciolisce per cui l'indice dell'array stampo non corrisponde più alla posizione della giusta A che intanto è slittata indietro di tante caselle quante sono le A già eliminate (che poi è il contatore_cancellare_RS)
                                    NW_j_t_liquid_assets_consumption[posizione_risultato] = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[azienda_i_capital] + NW_j_t_liquid_assets_consumption[posizione_risultato]
                                    contatore_cancellare_RS = 1 + contatore_cancellare_RS
        NW_i_t_liquid_assets_capital[azienda_i_capital] = NW_i_t_liquid_assets_capital[azienda_i_capital] - c_i_t_unit_cost_production[azienda_i_capital] * Q_i_t_quante_A_macchine_deve_produrre # nel caso Q_i_t è già stata aggiornata, non puoi usare c_i_t_Q_i_t poichè non era stato aggiornato dopo i controlli
        # I_j(t)= EI_j(t) + RI_j(t)
        # Per calcolare gli investimenti I totali, la parte totale dei RI e quella dei EI aggiorno ad ogni passaggio del loop sulle F2 aggiungendo la quota della azienda _i. Sia quantità che i valori di queste cioè i soldi. Ricordati che questi vettori sono storici con i vari turni come elementi, non sono liste ma np.array
        I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] = Q_i_t_quante_A_macchine_deve_produrre + I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno]
        I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] = Q_i_t_quante_A_macchine_deve_produrre * prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[azienda_i_capital] + I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno]
        RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] = np.sum(RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :]) + RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno]
        RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] = np.sum(RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :]) * prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[azienda_i_capital] + RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno]
        EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno] = Q_i_t_quante_A_macchine_deve_produrre - np.sum(RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :]) + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno]
        EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno] = (Q_i_t_quante_A_macchine_deve_produrre - np.sum(RI_quante_macchine_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j[azienda_i_capital, :]) )*prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[azienda_i_capital] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno]
        
        # Progresso tecnologico, poi a seconda se fai versione paper 2010 dovrà essere cambiato
        B_i_new = aziende_capital_good_B_i_tau[azienda_i_capital] * (1 + np.random.uniform(i_meno_2_quello_di_B_i_tau, i_piu_2_quello_di_B_i_tau) )
        A_i_new = aziende_capital_good_A_i_tau[azienda_i_capital] * (1 + np.random.uniform(i_meno_1_quello_di_A_i_tau, i_piu_1_quello_di_A_i_tau) ) # se scrivi np.random.uniform(-0.5, 0.5, 1) anche se è solo uno ti viene come risultato non uno scalare ma un np.array, se non metti l'indicazione di quanti ne vuoi, perciò metti solo gli estremi della distribuzione, viene uno scalare come serve a te
        if A_i_new > aziende_capital_good_A_i_tau[azienda_i_capital]:
            fake_innovazione_aziende_capital_good_A_i_tau[azienda_i_capital] = A_i_new # sovrascrivo poichè è una tecnologia migliore e la tengo
        if B_i_new > aziende_capital_good_B_i_tau[azienda_i_capital]:
            fake_innovazione_aziende_capital_good_B_i_tau[azienda_i_capital] = B_i_new
        
        # --- QUESTO E' STATO SPOSTATO FUORI PER RISOLVERE IL PROBLEMA DEL TURNO, FORSE SI PUO' ELIMINARE ----
        # calcolo il nuovo prezzo delle tecnologia A_i_tau venduta dalla azienda _i
        #c_i_t_unit_cost_production = w_salario_questo_turno/aziende_capital_good_B_i_tau[azienda_i_capital]
        # PROBLEMA! Siccome la brochure andrebbe mandata all'inizio del prossimo turno questo salario non dovrebbe essere quello di questo turno ma del prossimo!!!
        #p_i_t_prezzo_singola_merce_capital_good_machine = c_i_t_unit_cost_production # Nel 2006 usava il mark-up anche per le capital _i, invece nel 2008 (ma nel successivo no!) dice espressamente "The price p_i is equal to the unit cost of production". Nel 2010 invece usa nuovamente un mark-up che è una costante.
        # Qua dice che  p_i(t) = c_i(t) Quindi abbiamo che siccome B_i^tau = w(t)/c_i(t)  allora  L_i^D(t) = ( Q_i(t) c_i(t) )/w(t)  e quindi w(t) L_i^D(t) = Q_i(t) c_i(t)
        # Quindi, le capital firms _i non incassano nulla dalla vendita dei macchinari! Semplicemente ci ripagano il costo del lavoro (salario x #lavoratori occupati) che hanno anticipato SE IL SALARIO E' QUELLO CON CUI HANNO PRODOTTO E NON QUELLO DEL TURNO PRECEDENTE!
        # Nel 2010 invece usa nuovamente un mark-up che è una costante!
        # p_i_t_prezzo_singola_merce_capital_good_machine = (1 + mi_1_mark_up_capital_good_firms_rule) * c_i_t_unit_cost_production # nel paper 2010 parametro costante: mi_1_mark_up_capital_good_firms_rule=0.04 
        #prezzi_tutte_capital_firms_per_tecnologia_A_i_tau[azienda_i_capital] = p_i_t_prezzo_singola_merce_capital_good_machine
        # ---   ----
        
        # Le brochure ai nuovi clienti andrebbero mandate all'inizio del turno stando al paper, io invece lo faccio alla fine del turno perchè sennò dovrei anticipare il loop delle F2 prima di quello delle F1  poi farne un altro dopo!!!
        if len(clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_i_capital]) < F1_numero_consumption_industry:
            # siccome non puoi mandare più brochure di F1 e trovando tutti 1 nella riga _i della matrice continuerebbe ad estrarre nuovi indici casuali per "nuova_azienda_consumo" in loop all'infinito!!!
            # Nuovi clienti storici
            NC_i_t = (k_sample_coeff_nuovi_clienti) * valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_i_capital]
            # Nel paper francamente non si capisce se i nuovi clienti sono a caso per questo turno o se si aggiungono per sempre e poi non si capisce se NC è una percentuale o se è HC più la percentuale perciò poi i clienti totali sarebbero 2*HC + k*HC, vedi qua sotto che è l'eq del paper
            # NC_i_t = (1 + k_sample_coeff_nuovi_clienti) * valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_i_capital]
            quante_nuove = int(valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_i_capital] + NC_i_t) - int(valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_i_capital])
            if quante_nuove > 0:
                # così serve almeno una nuova consumption
                nuove_consumption_firms_mandare_brochure_indici = [] # tutte le consumption non ancora nella lista che potrebbero diventare nuovi clienti
                for contatore_colonna_per_spostamento_lungo_la_riga in range(F1_numero_consumption_industry):
                    if matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[azienda_i_capital, contatore_colonna_per_spostamento_lungo_la_riga] == 0:
                        nuove_consumption_firms_mandare_brochure_indici.append(contatore_colonna_per_spostamento_lungo_la_riga)
                if quante_nuove < len(nuove_consumption_firms_mandare_brochure_indici):
                    for quante_da_pescare in range(quante_nuove):
                        indice_estratto_da_eliminare = np.random.randint(0,len(nuove_consumption_firms_mandare_brochure_indici))
                        nuova_azienda_consumo = nuove_consumption_firms_mandare_brochure_indici.pop(indice_estratto_da_eliminare) #siccome pop non solo elimina l'elemento con quell'indice ma restituisce anche il suo valore
                        matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[azienda_i_capital, nuova_azienda_consumo] = 1
                        brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[nuova_azienda_consumo] = np.append(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[nuova_azienda_consumo], azienda_i_capital)
                        clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_i_capital] = np.append(clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_i_capital], nuova_azienda_consumo) # è la versione speculare di quello sopra
                else:
                    # cioè se quante_nuove >= len(nuove_consumption_firms_mandare_brochure_indici) le devo prendere tutte
                    for nuova_azienda_consumo in nuove_consumption_firms_mandare_brochure_indici:
                        matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[azienda_i_capital, nuova_azienda_consumo] = 1
                        brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[nuova_azienda_consumo] = np.append(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[nuova_azienda_consumo], azienda_i_capital)
                        clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_i_capital] = np.append(clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_i_capital], nuova_azienda_consumo) # è la versione speculare di quello sopra
            valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_i_capital] = NC_i_t + valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_i_capital] # aggiorno il valore che sarà probabilmente un numero con la virgola
        

    if turno>0 and restart_j_t_fallite_tutte == 0:
        # ora bisogna tenere conto anche se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente
        # solo dal turno=1 in poi credo che sia possibile calcolare la competitiveness E_j(t) che serve ora per calcolare l'average
        E_t_average_sectorial_consumption_competitiveness = np.dot( E_competitiveness_tutte_consumption_firms_t, f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1] ) # le f_j(t) sono percentuali per cui sono già divise per numero totale, che è F_1, per cui non hai bisogno di dividere per fare la media di E
        E_average_competitiveness_consumption_storico_tutti_turni = np.append(E_average_competitiveness_consumption_storico_tutti_turni, E_t_average_sectorial_consumption_competitiveness)
    for azienda_j_consumption in range(F1_numero_consumption_industry):
        # prima sostituire le macchine e aggiungere le nuove, ovvero RI e EI, poi distruggere, se vuoi, quelle troppo vecchie
        # che non sono state sostituite per mancanza di soldi o perchè le capital non avevano abbastanza disoccupati da assumere per produrre tutte le nuove macchine richieste
        # Sostituisco le macchine da sostituire
        for indice_contatore_posizione in range(len(RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption])):
            # utilizzo delle scorciatoie per semplificare la notazione che sennò sarebbe mastodontica
            dove_sta_la_macchina = RS_t_posizioni_tecn_A_da_sostituire_nel_vettore_consumption_firms[azienda_j_consumption][indice_contatore_posizione]
            con_quale_capital_macchina_sostituire = indici_capital_firms_con_cui_sostituire_le_A_in_consumption_firms[azienda_j_consumption][indice_contatore_posizione]
            aziende_consumption_good[azienda_j_consumption][dove_sta_la_macchina] = aziende_capital_good_A_i_tau[con_quale_capital_macchina_sostituire]
            eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption][dove_sta_la_macchina] = -1 # faccio ripartire i suoi anni
            # -1 invece che 0, così poi aggiungendo +1 vanno a zero altrimenti dovresti escludere le nuove appena aggiunte quando dopo aggiornerai l'età e sarebbe complicato
        # Aggiungo le nuove macchine per gli investimenti di espansione
        if EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption] > 0:
            con_quale_capital_macchine_tutte_uguali = EI_indici_capital_firms_con_cui_espandere_theta_in_consumption_firms[azienda_j_consumption]
            for quante_aggiungere_a_theta_j in range(EI_t_quante_nuove_tecn_A_consumption_firms_vuole_comprare[azienda_j_consumption]):
                aziende_consumption_good[azienda_j_consumption] = np.append(aziende_consumption_good[azienda_j_consumption], aziende_capital_good_A_i_tau[con_quale_capital_macchine_tutte_uguali])
                eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption] = np.append(eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption], -1)
        # PRIMA DISTRUGGI, SE VUOI (NEI PAPER NON SI CAPISCE SE LO FA O NO!), LE MACCHINE TROPPO VECCHIE E POI AGGIUNGI UN ANNO DI ETA'
        # ALTRIMENTI, SE LO FAI PRIMA, LE MACCHINE PIU' VECCHIE DI eta VERRANNO DISTRUTTE ORA E NON ARRIVERANNO AL PROSSIMO TURNO DOVE LA CONDIZIONE SUL LOOP DEGLI RI LE AVREBBE SOSTITUITE
        aziende_consumption_good[azienda_j_consumption] = np.delete(aziende_consumption_good[azienda_j_consumption], np.where(eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption] >= eta_max_machine_age))
        eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption] = np.delete(eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption], np.where(eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption] >= eta_max_machine_age))
        # Aggiorno l'età "eta" dei macchinari dell'azienda j a fine turno
        eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption] = 1 + eta_macchinari_vecchiaia_consumption_aziende[azienda_j_consumption]
        # funziona poichè lavorare sulla singola componente del vettore sostituendone la nuova versione aggiorna tutto il vettore, in questo caso le nuove vecchiaie compaiono nel vettore dei macchinari dell'azienda j
        # loop per calcolare le market share di ogni j poichè il bisogno di calcolare average E DOPO aver calcolato le singole E_j, ti ha costretto ad uscire dal loop precedente!
        if turno>0 and restart_j_t_fallite_tutte == 0:
        #if turno>0: # prima era così poi hai aggiunto la parte restart poichè
            # ora bisogna tenere conto anche se c'è il restart perchè tutte le _j sono morte alla fine del turno precedente
            # solo dal turno=1 in poi credo che sia possibile calcolare la market share f_j(t)
            # --- ORA, MIA SUPPOSIZIONE, POICHE' NEL PAPER SEMBRA CHE E_j(t) SIA NEGATIVA E QUINDI ANCHE LA MEDIA ---
            # Esplicitando che chi=-0.5 viene:  f_j(t) = f_j(t-1)*(1.5 - 0.5 * E_j(t)/E_bar(t))  quindi
            # E_j(t) è negativa e compresa tra  0 =< E_j(t) < -inf  per cui E_j(t)€[0,-inf[ abbiamo supposto che l=0 se Q_j(t-1)>=D_j(t-1) quindi E_j(t)=-p_j(t) e al limite p_j-->0 quindi al massimo si può avere E_j(t)=0
            # questa E_j(t) è una grande E_j(t) poichè tutte le E_j(t) saranno negative e quindi una E_j(t)=0>E_y(t) con y!=j siccome E_y(t)<=0
            # Però in valore assoluto una E_j(t)=-1 è < di una E_j(t)=-9: E_j(t)=-1>-9=E_y(t) ma |E_j(t)|=1<9=|E_y(t)| questo è importante  siccome E_j(t)/E_bar(t) è simile ad un |..| poichè i due segni meno si cancellano e quindi:
            # Se  |E_j(t)| > |E_bar(t)|  allora  f_j(t) < f_j(t-1)  ovvero f_j decresce rispetto al turno precedente poichè _j è MENO competitiva (E_j(t) è più lontana da zero rispetto al turno precedente)  e E_j(t)/E_bar(t)>1 e quindi toglie molto a 1,5-0,5*1,..=1,5-0,5.. così f_j(t-1)*0,..
            # Se  |E_j(t)| < |E_bar(t)|  allora  f_j(t) > f_j(t-1)  ovvero f_j cresce rispetto al turno precedente poichè _j è PIU' competitiva (E_j(t) è più vicina a zero rispetto al turno precedente)  e E_j(t)/E_bar(t)<1 e quindi toglie poco a 1,5-0,5*0,..=1,5-0,4.. così f_j(t-1)*1,..
            # Se  E_j(t) = E_bar(t)  allora  f_j(t) = f_j(t-1)  ovvero f_j rimane uguale al turno precedente    e E_j(t)/E_bar(t)=1 e quindi 1,5-0,5*1=1,5-0,5=1,00 così f_j(t-1)*1=f_j(t)
            # L'incremento massimo che f_j(t) può avere rispetto a f_j(t-1) è di 0.5 che corrisponde a una E_j(t)=0 cioè massima,  per cui si ha  f_j(t)=f_j(t-1)*(1.5 - 0.5* 0/E_bar(t))=f_j(t-1)*(1.5 - 0)=f_j(t-1) * 1.5=f_j(t-1) + 0.5*f_j(t-1)
            f_j_t_market_share_consumption = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1][azienda_j_consumption] * (1 + chi_replicator_dynamics_coefficient * ( (E_competitiveness_tutte_consumption_firms_t[azienda_j_consumption] - E_t_average_sectorial_consumption_competitiveness)/(E_t_average_sectorial_consumption_competitiveness) ) )
            # +++---+++---+++
            # NUOVA PARTE PER RISOLVERE BUG GRANDEZZE NEGATIVE
            if f_j_t_market_share_consumption <= 0:
                # Infatti il problema è che se f_j(t) è negativa poi dopo, più avanti, anche la domanda D_j(t) per la vendita verrà negativa e a ruota anche le altre grandezze
                f_j_t_market_share_consumption = 0
            # +++---+++---+++
            f_market_share_tutte_consumption_firms_storico_tutti_turni[turno] = np.append(f_market_share_tutte_consumption_firms_storico_tutti_turni[turno], f_j_t_market_share_consumption) # al vettore np.array del turno (creato prima vuoto) aggiungo il valore di f della azienda j


    # SE LE MARKET SHARE f_j(t) NON DOVESSERO SOMMARE a 1 cioè 100% puoi riscalarle affinchè le nuove market share sommino all'unità Sum_{j=0}^{F1-1} f'_j(t)=1=100% dove f'_j(t)=f_j(t)/( Sum_{j=0}^{F1-1} f_j(t) )
    f_market_share_tutte_consumption_firms_storico_tutti_turni[turno] = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno]/np.sum(f_market_share_tutte_consumption_firms_storico_tutti_turni[turno])


    # CONSUMI e PROFITTI AZIENDE
    Emp_t_lavoratori_occupati = L_0_labor_supply_popolazione_lavoratori - persone_ancora_disoccupate # nell'ipotesi però che la popolazione non aumenti nel tempo!
    # work-or-die scenario
    C_t_valore_monetario_consumi_rivolti_alle_consumption = w_salario_questo_turno * Emp_t_lavoratori_occupati
    # social-security scenario
    # C_t_valore_monetario_consumi_rivolti_alle_consumption = w_salario_questo_turno *( Emp_t_lavoratori_occupati + phi_wage_share_sussidio_rispetto_salario * persone_ancora_disoccupate ) # se c'è sussidio di disoccupazione
    D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno] * C_t_valore_monetario_consumi_rivolti_alle_consumption # ovviamente: Sum D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita = C_t_valore_monetario_consumi_rivolti_alle_consumption
    D_j_t_domanda_questo_turno_quantita = np.zeros(F1_numero_consumption_industry)
    D_j_t_domanda_questo_turno_quantita = np.around( D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita / prezzi_tutte_consumption_firms_p_j_t ) # round non funziona su np.array.
    for azienda_j_consumption in range(F1_numero_consumption_industry):
        if D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] >= Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption]:
            # consuma le riserve dei magazzini siccome non si è prodotto abbastanza
            quanto_non_ha_prodotto = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] - Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption] # può essere zero se D_j_t=Q_j_t
            if quanto_non_ha_prodotto < N_j_turno_precedente_merce_invenduta[azienda_j_consumption]:
                N_j_turno_precedente_merce_invenduta[azienda_j_consumption] = np.round( N_j_turno_precedente_merce_invenduta[azienda_j_consumption] - quanto_non_ha_prodotto )
                S_j_t_attuale_total_sales_soldi[azienda_j_consumption] = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] * prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption]
                S_j_turno_precedente_total_sales_soldi[azienda_j_consumption] = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] * prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption]
            else:
                # ovvero caso quanto_non_ha_prodotto >= N_j_turno_precedente_merce_invenduta[azienda_j_consumption]
                S_j_t_attuale_total_sales_soldi[azienda_j_consumption] = (Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption] + N_j_turno_precedente_merce_invenduta[azienda_j_consumption]) * prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption]
                S_j_turno_precedente_total_sales_soldi[azienda_j_consumption] = (Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption] + N_j_turno_precedente_merce_invenduta[azienda_j_consumption]) * prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption]
                N_j_turno_precedente_merce_invenduta[azienda_j_consumption] = 0 # poichè i magazzini vengono svuotati ma non basta
                # Non si può ridistribuire quella quantità che manca ad una altra _j siccome i prodotti di quella _j avrebbero un altro prezzo e la quantità mancante sarebbe differente magari anche <1
                # quindi semplicemente quei soldi del salario per cui non si può comprare niente spariscono dalla simulazione!
                # Oppure no, in questo modo si può fare:
                # +++ PER RIDISTRIBUIRE DOMANDA INSODDISFATTA DALLA azienda_j_consumption, CHE NON HA ABBASTANZA SCORTE NEI MAGAZZINI, AD UN ALTRA _j, PERO' SENZA CAMBIARE LE f_j(t) DI ENTRAMBE +++
                # if azienda_j_consumption < (F1_numero_consumption_industry - 2):
                    # poichè range(F1) va da 0 a F1-1 ovvero da 0 a 199 poichè F1=200 e nella realtà partirebbe da 1 e arriverebbe a 200 incluso, invece che partire da zero
                    # controllo che ci sia almeno un'altra consumption a cui passare la domanda insoddisfatta da questa _j
                    # lista_estrazione_indici_j_cui_passare_domanda_insoddisfatta = [ci for ci in range(1 + azienda_j_consumption, F1_numero_consumption_industry)] # COME ERA PRIMA, PRIMA DI SCOPRIRE IL BUG
                    # +++---+++---+++
                    # lista_estrazione_indici_j_cui_passare_domanda_insoddisfatta = []
                    # for ci in range(1 + azienda_j_consumption, F1_numero_consumption_industry):
                        # if f_market_share_tutte_consumption_firms_storico_tutti_turni[turno][ci] > 0:
                            # lista_estrazione_indici_j_cui_passare_domanda_insoddisfatta.append(ci)
                    # if len(lista_estrazione_indici_j_cui_passare_domanda_insoddisfatta) > 0:
                    # +++---+++---+++
                        # np.random.shuffle(lista_estrazione_indici_j_cui_passare_domanda_insoddisfatta) # non crea una nuova lista ma permuta casualmente i membri della lista
                        # nuova_j_estratta = lista_estrazione_indici_j_cui_passare_domanda_insoddisfatta[0] # COME ERA PRIMA, PRIMA DI SCOPRIRE IL BUG
                        # D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita[nuova_j_estratta] = (D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita[azienda_j_consumption] - S_j_t_attuale_total_sales_soldi[azienda_j_consumption]) + D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita[nuova_j_estratta]
                        # D_j_t_domanda_questo_turno_quantita[nuova_j_estratta] = int( np.round( D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita[nuova_j_estratta]/prezzi_tutte_consumption_firms_p_j_t[nuova_j_estratta] ) )
                # +++ +++
                # +++ ORA QUA CAMBIO INVECE LE f_j(t) DI ENTRAMBI AGGIORNANDOLE, se non vuoi farlo basta commentare +++
                        #f_market_share_tutte_consumption_firms_storico_tutti_turni[turno][nuova_j_estratta] = D_j_t_domanda_questo_turno_soldi_prezzi_x_quantita[nuova_j_estratta]/C_t_valore_monetario_consumi_rivolti_alle_consumption
                        #f_market_share_tutte_consumption_firms_storico_tutti_turni[turno][azienda_j_consumption] = S_j_t_attuale_total_sales_soldi[azienda_j_consumption]/C_t_valore_monetario_consumi_rivolti_alle_consumption
                        # COSI' FACENDO, POTREBBE ESSERE CHE ORA LE MARKET SHARE f_j(t) NON SOMMINO a 1 cioè 100% puoi riscalarle affinchè le nuove market share sommino all'unità Sum_{j=0}^{F1-1} f'_j(t)=1=100% dove f'_j(t)=f_j(t)/( Sum_{j=0}^{F1-1} f_j(t) )
                        #f_market_share_tutte_consumption_firms_storico_tutti_turni[turno] = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno]/np.sum(f_market_share_tutte_consumption_firms_storico_tutti_turni[turno])
                # +++ +++
        else:
            # ovvero caso D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] < Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption]
            # ciò che avanza poichè invenduto finirà nelle scorte, cioè ha prodotto troppo
            quanto_in_piu_prodotto = Q_j_t_quantita_prodotta_nel_turno_attuale[azienda_j_consumption] - D_j_t_domanda_questo_turno_quantita[azienda_j_consumption]
            N_j_turno_precedente_merce_invenduta[azienda_j_consumption] = np.round( N_j_turno_precedente_merce_invenduta[azienda_j_consumption] + quanto_in_piu_prodotto )
            S_j_t_attuale_total_sales_soldi[azienda_j_consumption] = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] * prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption]
            S_j_turno_precedente_total_sales_soldi[azienda_j_consumption] = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] * prezzi_tutte_consumption_firms_p_j_t[azienda_j_consumption]
        D_j_domanda_turno_precedente[azienda_j_consumption] = D_j_t_domanda_questo_turno_quantita[azienda_j_consumption] # ora che non mi serve più aggiorno per il prossimo turno
        Pi_j_t_profitti_parziali = S_j_t_attuale_total_sales_soldi[azienda_j_consumption] - r_interest_rate * Deb_j_turno_precedente_debito_consumption[azienda_j_consumption] # parte dei profitti, ovvero il costo della produzione c_j(t)Q_j(t) è già stato sottratto prima direttamente ai NW_j(t) liquid asset
        NW_j_t_liquid_assets_consumption[azienda_j_consumption] = Pi_j_t_profitti_parziali + NW_j_t_liquid_assets_consumption[azienda_j_consumption] # invece cI_j cioè "amount of internal funds" erano già stati sottratti dai NW_j(t) durante la parte sui EI e RI
    # Ora bisogna dare alle capital _i il denaro incassato dalla vendita delle macchine A alle _j
    NW_i_t_liquid_assets_capital = NW_i_t_liquid_assets_capital + numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption * prezzi_tutte_capital_firms_per_tecnologia_A_i_tau
    # NW_i_t_liquid_assets_capital = NW_i_t_liquid_assets_capital + prezzi_tutte_capital_firms_per_tecnologia_A_i_tau * np.sum(quante_macchine_totali_matrice_F2xF1_ordinate_alla_riga_i_dalla_colonna_j, 1)
    # nel caso dovessi eliminare dal programma "numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption" poichè ridondante, qua per farne a meno. L'1 serve per sommare sulle righe della matrice, sennò con 0 somma sulle colonne e senza, invece, somma ogni numero nella matrice!
    


    N_j_copiato_per_recuperare_merce_invenduta_prima_che_aziende_j_falliscano = np.copy(N_j_turno_precedente_merce_invenduta) # servirà alla fine per calcolare i magazzini DN, così ora puoi modificare tranquillamente il vettore N_j e mettere a 0 le _j che falliscono tanto hai questa e la copia iniziale che è tutto ciò che ti serve per calcolare DN_j
    produzione_effettiva_tutte_consumption_storico_tutti_turni = np.append(produzione_effettiva_tutte_consumption_storico_tutti_turni, np.sum(Q_j_t_quantita_prodotta_nel_turno_attuale) )
    produzione_massima_tutte_consumption_storico_tutti_turni = np.append(produzione_massima_tutte_consumption_storico_tutti_turni, np.sum(np.sum(falsi) for falsi in aziende_consumption_good) )
    # Scarico il segnalatore dei fallimenti di tutte le aziende e preparo quello del turno successivo poichè servirà per non calcolare il mark up delle _j e prendere quello iniziale
    if restart_j_t_fallite_tutte == 1:
        restart_j_t_fallite_tutte = 0
        passato_solo_turno_dal_restart_mark_up = 1
    else:
        # cioè se restart_j_t_fallite_tutte == 0
        restart_j_t_fallite_tutte = 0 # continuo a tenerlo a zero
        passato_solo_turno_dal_restart_mark_up = 0
    #restart_j_t_fallite_tutte = 0 # se non servisse "passato_solo_turno_dal_restart_mark_up", alternativa a come fatto sopra tanto non serve l'if poichè per entrambi i casi fai la stessa cosa per questa variabile cioè la poni uguale a zero
    restart_i_t_fallite_tutte = 0 # per quello delle _i dove non c'è bisogno di un accorgimento per il mark up come per le _j, basta metterlo a zero.
    # La stessa cosa si poteva fare anche sopra per le _j senza if. Inoltre se ci sarà un fallimento totale in questo turno verrano cambiati successivamente nelle prossime righe.

    
    # EXIT and ENTRY: chi muore e chi sopravvive
    # AZIENDE CONSUMPTION _j (F1)
    # Secondo me non ha senso uccidere le aziende consumption nel primo turno poichè all'inizio non puoi calcolare f_j(t) siccome per quella serve (t-1) ed essendo t=0 servirebbe t=-1, poi hanno tutte gli stessi valore di A e gli stessi prezzi
    if turno > 0:
        soglia_sotto_alla_quale_consumption_j_muore = (1/F1_numero_consumption_industry)*(1/10) # CON QUESTO HAI FATTO TUTTE LE SIMULAZIONI
        # soglia_sotto_alla_quale_consumption_j_muore = (1/F1_numero_consumption_industry)*(1/3) # con questo hai fatto tutte simulazioni scala_rov fissa e variabile. E' importante siccome copi i magazzini delle _j morte e più ne muoiono più si saturano
        # 1/F1 è se tutte avessero la stessa quota di mercato, cioè f_j(t)*F1=(1/F1)*F1=1. Non voglio che una consumption _j sia dichiarata morta solo nel caso abbia f_j(t)=0 ma che sia "quasi" zero come dice nel paper, ovvero io ho scelto arbitrariamente 100 volte più piccola di questa quota
        lista_estrazione_indici_j_non_fallite = []
        lista_estrazione_indici_j_fallite = []
        for azienda_j_consumption in range(F1_numero_consumption_industry):
            # Risolvo il problema delle _j che non hanno ricevuto brochure non possono sostituire le macchine e quando queste muoino si ritrovano l'array vuoto e dopo K=0 e quando viene fatta la divisione per K il programma si arresta poichè /0
            # allora o faccio in modo che tutte le j abbiano almeno una brochure da cui ordinare oppure uccido artificialmente una _j che non ha più macchine, scelgo la seconda
            if len(aziende_consumption_good[azienda_j_consumption]) == 0:
                NW_j_t_liquid_assets_consumption[azienda_j_consumption] = -5000111000 # così la faccio morire manualmente poichè ha asset negativi
            # Il problema è che io prima di valutare se una _j è fallita oppure no devo conoscere sia quali _j non sono fallite, per potere assegnarne una non fallita (cioè copiarne i dati) alla _j fallita, che quante non sono fallite, siccome, se tutte quante sono fallite, devo fare restart di tutte le _j, che è quasi come reiniziare la simulazione!
            if f_market_share_tutte_consumption_firms_storico_tutti_turni[turno][azienda_j_consumption] > soglia_sotto_alla_quale_consumption_j_muore and NW_j_t_liquid_assets_consumption[azienda_j_consumption] > 0:
                # serve "and" poichè è sufficiente che una delle due sia False perchè l'azienda _j sia fallita. Infatti stai cercando se non è fallita. Se invece valutassi se è fallita servirebbe or poichè è sufficiente che una delle due sia True ovvero ci sia una condizione del fallimente per dichiararla fallita. 
                lista_estrazione_indici_j_non_fallite.append(azienda_j_consumption)
            else:
                lista_estrazione_indici_j_fallite.append(azienda_j_consumption)
                quante_fallite_j_aziende_questo_turno_storico_tutti_turni[turno] = 1 + quante_fallite_j_aziende_questo_turno_storico_tutti_turni[turno]
                numero_fallimenti_ogni_consumption_j[azienda_j_consumption] += 1
        if len(lista_estrazione_indici_j_fallite) > 0:
            # poichè se nessuna _j è fallita non ha senso andare avanti
            if len(lista_estrazione_indici_j_non_fallite) > 0:
                # poichè ci devono essere delle _j non fallite da usare come stampo per quelle fallite
                # non si può usare solo questa condizione, poichè il fatto che non vi sia nessuna _j non fallita, non significa che le _j siano tutte vive: ce ne potrebbero essere molte fallite e qualcuna no
                for azienda_fallita_j_consumption in lista_estrazione_indici_j_fallite:
                    indice_estratto_consumption_da_clonare = np.random.choice(lista_estrazione_indici_j_non_fallite)
                    aziende_consumption_good[azienda_fallita_j_consumption] = np.copy(aziende_consumption_good[indice_estratto_consumption_da_clonare])
                    # ATTENTO che siccome sono np.array non puoi usare solo = ma devi usare np.copy() poichè sennò quando modifichi la copia cambia anche l'originale e viceversa poichè in realtà sono lo stesso, è come se fosse un puntatore!!!
                    # Questo se tenti di copiare un intero vettore, tipo vettore1=vettore2 invece se fai vettore1[posizione]=vettore2[posizione] non ci sono problemi, però qua era necessario usare copy perchè sono vettori di vettori quindi vettore1[posizione] e vettore2[posizione] sarebbero a loro volta vettori e non scalari!
                    eta_macchinari_vecchiaia_consumption_aziende[azienda_fallita_j_consumption] = np.copy(eta_macchinari_vecchiaia_consumption_aziende[indice_estratto_consumption_da_clonare]) # solo per avere la stessa dimensione poichè sennò avrebbe un numero diverso di macchine A ovvero un vettore Theta_j di lunghezza differente non giusta
                    # devi far così se vuoi mettere a zero l'intero np.array di una lista altrimenti se non metti la seconda [] con [:] ti sostituisce con un solo 0 quell'array e non è neanche un np.array 
                    # QUESTA RIGA SOTTO ERA ATTIVA!!!111!11111 L'HAI TOLTA PER ALLINEARE LE _j CON LA REGOLA DEGLI RI ogni b ANNI:
                    eta_macchinari_vecchiaia_consumption_aziende[azienda_fallita_j_consumption][:] = 0 # faccio ripartire gli anni però non metto a -1 poichè il +1 è già stato aggiunto per le nuove macchine
                    # Deb_j_turno_precedente_debito_consumption[azienda_fallita_j_consumption] = Deb_j_turno_precedente_debito_consumption[indice_estratto_consumption_da_clonare]
                    Deb_j_turno_precedente_debito_consumption[azienda_fallita_j_consumption] = 0
                    # NW_j_t_liquid_assets_consumption[azienda_fallita_j_consumption] = NW_i_j_0_liquid_assets
                    # NW_j_t_liquid_assets_consumption[azienda_fallita_j_consumption] = NW_j_t_liquid_assets_consumption[indice_estratto_consumption_da_clonare]
                    NW_j_t_liquid_assets_consumption[azienda_fallita_j_consumption] = scala_lavoratori_capital_good_parametro_mail_roventini * NW_i_j_0_liquid_assets * w_salario_questo_turno
                    # NW_j_t_liquid_assets_consumption[azienda_fallita_j_consumption] = parametro_mail_roventini_scala_lavoratori * NW_i_j_0_liquid_assets * w_salario_questo_turno
                    # NW_j_t_liquid_assets_consumption[azienda_fallita_j_consumption] = w_salario_questo_turno * NW_i_j_0_liquid_assets
                    # come brochure arrivate a _j tengo le stesse della morente quindi non copio quelle dell'indice estratto,
                    N_j_turno_precedente_merce_invenduta[azienda_fallita_j_consumption] = 0
                    # N_j_turno_precedente_merce_invenduta[azienda_fallita_j_consumption] = N_j_turno_precedente_merce_invenduta[indice_estratto_consumption_da_clonare]
                    # matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[:, azienda_fallita_j_consumption] = np.copy(matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[:, indice_estratto_consumption_da_clonare]) # COMUNQUE NON SERVE POICHE' TIENI LE STESSE BROCHURE CHE AVEVA LA MORTA, ALTRIMENTI DOVRESTI AGGIORNARE ANCHE LE CAPITAL _i
                    # va bene poichè bisogna copiare una colonna in un'altra colonna e [:,_j] restituisce la colonna come np.array. Forse non ci sarebbe neanche bisogno di usare np.copy ma basterebbe solo = perchè l'array restituito non è lo stesso, però per sicurezza meglio usarlo
                    mi_mark_up_tutte_consumption_firms_storico_tutti_turni[turno][azienda_fallita_j_consumption] = mi_mark_up_tutte_consumption_firms_storico_tutti_turni[turno][indice_estratto_consumption_da_clonare]
                    f_market_share_tutte_consumption_firms_storico_tutti_turni[turno][azienda_fallita_j_consumption] = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno][indice_estratto_consumption_da_clonare]
                    f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1][azienda_fallita_j_consumption] = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1][indice_estratto_consumption_da_clonare]
                    # poichè per calcolare il mark-up del prossimo turno, dell'azienda fallita risorta come clone di una viva estratta, servirà il mark-up del turno precedente e le market share degli ultimi due turni, se non le copiassi come avevo pensato di fare all'inizio, poi la nuova azienda sarebbe avvantaggiata e magari salterebbero fuori altri comportamenti strani non previsti dal modello
                    # S_j_turno_precedente_total_sales_soldi[azienda_fallita_j_consumption] = S_j_turno_precedente_total_sales_soldi[indice_estratto_consumption_da_clonare]
                    # Q_j_quantita_prodotta_turno_precedente[azienda_fallita_j_consumption] = Q_j_quantita_prodotta_turno_precedente[indice_estratto_consumption_da_clonare]
                    # D_j_domanda_turno_precedente[azienda_fallita_j_consumption] = D_j_domanda_turno_precedente[indice_estratto_consumption_da_clonare]
                    # Se vuoi fare in modo che l_j venga zero e quindi la nuova risorta non rischi una diminuzione immotivata della sua f_j subito, Q_j(t-1) e D_j(t-1) devono essere uguali. Il problema è che D_j(t-1) è artificiosamente impostata per avere EI appena risorta:
                    Q_j_quantita_prodotta_turno_precedente[azienda_fallita_j_consumption] = np.round( (1+alpha_trigger_rule) * len(aziende_consumption_good[indice_estratto_consumption_da_clonare]) * (np.sum(aziende_consumption_good[indice_estratto_consumption_da_clonare])/len(aziende_consumption_good[indice_estratto_consumption_da_clonare])) * (u_desired_level_capacity_utilization/(1+theta_parameter_desired_inventories)) )
                    D_j_domanda_turno_precedente[azienda_fallita_j_consumption] = np.round( (1+alpha_trigger_rule) * len(aziende_consumption_good[indice_estratto_consumption_da_clonare]) * (np.sum(aziende_consumption_good[indice_estratto_consumption_da_clonare])/len(aziende_consumption_good[indice_estratto_consumption_da_clonare])) * (u_desired_level_capacity_utilization/(1+theta_parameter_desired_inventories)) ) # tolto int64 per evitare il problema dell'overflow che np.sum di D_exp dava un numero negativo
                    # Q_j_quantita_prodotta_turno_precedente[azienda_fallita_j_consumption] = np.round( (1+alpha_trigger_rule) * len(aziende_consumption_good[indice_estratto_consumption_da_clonare]) * (np.sum(aziende_consumption_good[indice_estratto_consumption_da_clonare])/len(aziende_consumption_good[indice_estratto_consumption_da_clonare])) * (1/(1+theta_parameter_desired_inventories)) ) #PROVA SENZA u QUINDI 100% INVECE DI 75%
                    # D_j_domanda_turno_precedente[azienda_fallita_j_consumption] = np.round( (1+alpha_trigger_rule) * len(aziende_consumption_good[indice_estratto_consumption_da_clonare]) * (np.sum(aziende_consumption_good[indice_estratto_consumption_da_clonare])/len(aziende_consumption_good[indice_estratto_consumption_da_clonare])) * (1/(1+theta_parameter_desired_inventories)) ) #PROVA SENZA u QUINDI 100% INVECE DI 75%
                    # In realtà se theta=alpha e viene posto N_j(t-1)=0 poi di fatto la condizione iniziale per la domanda diventa u^d che qua è 75% della quantità massima che la nuova azienda _j risorta può produrre, cioè la somma delle sue A, tutto il resto si semplifica!
                    # Nel paper non dice come vada fatto, prima avevi fatto così:
                    # D_j_domanda_turno_precedente[azienda_fallita_j_consumption] = 0
                    # Q_j_quantita_prodotta_turno_precedente[azienda_fallita_j_consumption] = 0
                    S_j_turno_precedente_total_sales_soldi[azienda_fallita_j_consumption] = 0
                # Bisogna riscalare le market share poichè sono cambiate a causa di avere copiato l'azienda; è sulla stessa riga di "for azienda_fallita_j_consumption in lista_estrazione_indici_j_fallite:" sopra.
                f_market_share_tutte_consumption_firms_storico_tutti_turni[turno] = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno]/np.sum(f_market_share_tutte_consumption_firms_storico_tutti_turni[turno])
                f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1] = f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1]/np.sum(f_market_share_tutte_consumption_firms_storico_tutti_turni[turno-1])
            else:
                # riferito a "if len(lista_estrazione_indici_j_non_fallite) > 0:"
                # Restart poichè tutte le consumption _j sono fallite e quindi non è possibile prenderne nessuno come stampo da copiare
                restart_j_t_fallite_tutte = 1
                contatore_restart_fallite_consumption_storico_tutti_turni += 1
                # Qua penseresti che ci sia il bisogno di fare il restart di tante altre variabili, invece mancano perchè, se le modificassi ora, i valori di questo turno andrebbero persi. Veranno messe invece all'inizio del turno
                #A_average_labor_productivity_solo_consumption_per_restart = ( np.sum( np.sum(aziende_consumption_good[falso]) for falso in range(F1_numero_consumption_industry) ) )/( np.sum(len(fake) for fake in aziende_consumption_good) ) # affinchè funzioni devi iterare i 2 loop come fai ora qua
                #aziende_consumption_good = [A_average_labor_productivity_solo_consumption_per_restart * np.ones(K_j_0_capital_stock_iniziale_per_azienda_beni) for ci in range(F1_numero_consumption_industry)] # è il vettore con i Theta_j_(t)  # Tecnicamente, è una lista di alcuni np.array
                # queste 2 righe sopra non vanno bene se il restart delle _j avviene proprio nel turno in cui tutti i macchinari morirebbero di vecchiaia perchè poi la lista di ciascuna _j è vuota e A_average viene np.nan e tutta la nuova lista di array delle consumption rinate viene quindi piena di NaN.
                aziende_consumption_good = [ A_average_productivity_solo_tutte_consumption_macchine[turno] * np.ones(K_j_0_capital_stock_iniziale_per_azienda_beni) for ci in range(F1_numero_consumption_industry) ] # è il vettore con i Theta_j_(t)  # Tecnicamente, è una lista di alcuni np.array
                eta_macchinari_vecchiaia_consumption_aziende = [np.zeros(K_j_0_capital_stock_iniziale_per_azienda_beni, dtype=int) for _ in range(F1_numero_consumption_industry)] # è una lista di un numero F1 di np.array, come la lista sopra
    # AZIENDE CAPITAL _i (F2)
    # Secondo me non ha senso uccidere le aziende capital nei primi turni poichè all'inizio hanno tutte lo stesso valore di A e B e questo cambia solo nel turno successivo, quindi almeno dovrà essere turno>1 
    # numero_turni_sopra_i_quali_ha_senso_quota_mercato = 2
    numero_turni_sopra_i_quali_ha_senso_quota_mercato = 1
    valore_della_singola_quota_mercato_capital_macchine_prodotte_questo_turno = prezzi_tutte_capital_firms_per_tecnologia_A_i_tau * numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption # è un np.array di lunghezza F2
    valore_dell_intera_quota_mercato_capital_macchine_prodotte_questo_turno = np.sum(valore_della_singola_quota_mercato_capital_macchine_prodotte_questo_turno) # è uno scalare, serve da denominatore della frazione per calcolare la quota f_i(t)
    if int(np.sum(numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption)) != 0:
        #f_market_share_tutte_capital_firms = valore_della_singola_quota_mercato_capital_macchine_prodotte_questo_turno/valore_dell_intera_quota_mercato_capital_macchine_prodotte_questo_turno # np.array di lunghezza F2
        # Forse però così è sbagliato poichè se guardi al valore monetario della quota di mercato, migliori sono le _i, più bassi sono i loro prezzi per cui a parità di macchine vendute una _i migliore ha una quota di mercato più piccola a causa del prezzo più basso. Un'alternativa, invece, potrebbe essere guardare alle quantità:
        #valore_dell_intera_quota_mercato_capital_macchine_prodotte_questo_turno = np.sum(numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption) # è uno scalare, è il numero totale di macchine costruite da tutte le _i su ordine delle _j, serve da denominatore della frazione per calcolare la quota f_i(t)
        f_market_share_tutte_capital_firms = numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption/np.sum(numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption) # np.array di lunghezza F2
        # Poi forse però devi modificare la soglia che deve essere un numero intero ora, ad esempio: soglia_sotto_alla_quale_capital_i_muore=1 oppure una percentuale del numero totale di macchine vendute, che è uguale a come fatto sopra cioè 1/(F2*10 o *100) ...
        # Così invece sarebbe sbagliato: #soglia_sotto_alla_quale_capital_i_muore = np.sum(numero_macchine_ogni_capital_i_deve_produrre_per_ordini_consumption)*(1/100)
    else:
        # cioè se nessuna _j ha comprato nessuna macchina A
        f_market_share_tutte_capital_firms = np.zeros(F2_numero_capital_industry)
    if turno > numero_turni_sopra_i_quali_ha_senso_quota_mercato:
        soglia_sotto_alla_quale_capital_i_muore = (1/F2_numero_capital_industry)
        soglia_sotto_alla_quale_capital_i_muore = (1/F2_numero_capital_industry)*(1/100) # 1/(F2*100)  Come è per le consumption, stessa soglia
        # 1/F2 è se tutte avessero la stessa quota di mercato, cioè f_i(t)*F2=(1/F2)*F2=1. Non voglio che una capital _i sia dichiarata morta solo nel caso abbia f_i(t)=0 ma che sia "quasi" zero come dice nel paper, ovvero io ho scelto arbitrariamente 100 volte più piccola di questa quota
        # Se invece guardi alle quantità e NON al valore monetario della quota di mercato (che sono le quantità moltiplicate per i loro prezzi). Però se dividi le quantità per la quantità totale di macchine vendute riottieni un percentuale: ed infatti non cambia niente!
        # Se hanno prodotto tutte quante lo stesso numero di macchine riottieni che ogni f_i=1/F2 infatti f_i=(macchinetot/F2)/macchinetot=1/F2.
        # L'unica cosa è che in generale forse per le capital è tanto avere F2*100, forse sarebbe meglio avere F2*10. Infatti pensa al caso in cui la macchinetot=500 e F2=10. Se come macchine vendute abbiamo 10: 10/500=0.02 e quindi 1/10*100=0.001 e 1/10*10=0.01. Non morirebbe. Se invece fosse 5: 5/500=0.01 e con 1: 1/500=0.002. Quindi, se come soglia scegliamo 0.001=1/10*100 morirebbero solo quelle _i che ne hanno vendute zero!
        lista_estrazione_indici_i_non_fallite = []
        lista_estrazione_indici_i_fallite = []
        for azienda_i_capital in range(F2_numero_capital_industry):
            quota_mercato_capital_i = f_market_share_tutte_capital_firms[azienda_i_capital]
            if quota_mercato_capital_i <= soglia_sotto_alla_quale_capital_i_muore or NW_i_t_liquid_assets_capital[azienda_i_capital] <= 0:
                # qua facciamo l'incontrario di come abbiamo fatto prima sopra per le consumption _j ed infatti c'è or invece che and. Questo è il caso dove _i è fallita
                lista_estrazione_indici_i_fallite.append(azienda_i_capital)
                quante_fallite_i_aziende_questo_turno_storico_tutti_turni[turno] = 1 + quante_fallite_i_aziende_questo_turno_storico_tutti_turni[turno]
                numero_fallimenti_ogni_capital_i[azienda_i_capital] += 1
            else:
                lista_estrazione_indici_i_non_fallite.append(azienda_i_capital)
        if len(lista_estrazione_indici_i_fallite) > 0:
            # poichè se nessuna _i è fallita non ha senso andare avanti
            if len(lista_estrazione_indici_i_non_fallite) > 0:
                # poichè ci devono essere delle _i non fallite da usare come stampo per quelle fallite
                # non si può usare solo questa condizione, poichè il fatto che non vi sia nessuna _i non fallita, non significa che le _i siano tutte vive: ce ne potrebbero essere molte fallite e qualcuna no
                for azienda_fallita_i_capital in lista_estrazione_indici_i_fallite:
                    indice_estratto_capital_da_clonare = np.random.choice(lista_estrazione_indici_i_non_fallite)
                    aziende_capital_good_A_i_tau[azienda_fallita_i_capital] = aziende_capital_good_A_i_tau[indice_estratto_capital_da_clonare]
                    aziende_capital_good_B_i_tau[azienda_fallita_i_capital] = aziende_capital_good_B_i_tau[indice_estratto_capital_da_clonare]
                    # non copio però l'innovazione perchè quella è segreta dell'azienda e non è stata sviluppata da quella entrante. Infatti ho messo nell'innovazione lo stesso valore sopra e non quello fake_innovazione_..ecc
                    fake_innovazione_aziende_capital_good_A_i_tau[azienda_fallita_i_capital] = aziende_capital_good_A_i_tau[azienda_fallita_i_capital]
                    fake_innovazione_aziende_capital_good_B_i_tau[azienda_fallita_i_capital] = aziende_capital_good_B_i_tau[azienda_fallita_i_capital]
                    matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[azienda_fallita_i_capital,:] = 0 # poichè i clienti storici della _i morente sono cancellati
                    for consumption_j_da_cancellare_a_cui_fallita_mandava_brochure in clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_fallita_i_capital]:
                        # devo cercare una a una le consumption _j alle quali _i mandava la brochure, poichè devo cancellare _i dai loro np.array nella lista delle brochure giunte a loro, essendo _i fallita
                        brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[consumption_j_da_cancellare_a_cui_fallita_mandava_brochure] = np.delete(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[consumption_j_da_cancellare_a_cui_fallita_mandava_brochure], np.where(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[consumption_j_da_cancellare_a_cui_fallita_mandava_brochure] == azienda_fallita_i_capital))
                        # np.delete ha bisogno dell'indice o di una lista di indici (cioè le posizioni) degli elementi che vuoi cancellare dall'np.array, ma tu non li sai però sai quali sono questi valori da cercare ovvero l'indice di _i,
                        # quindi con np.where restituisci le posizioni dove l'np.array assume quei valori e poi li passa a np.delete che li elimina
                    clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_fallita_i_capital] = np.array([], int) # cancello la vecchia lista
                    # clienti_storici_con_cui_inizializzare_nuova_entrante = np.random.randint(0, F1_numero_consumption_industry, 2*HC_i_0_clienti_iniziali) # suppongo sia il doppio dei clienti iniziali affinchè l' _i entrante non sia troppo svantaggiata
                    clienti_storici_con_cui_inizializzare_nuova_entrante = np.random.randint(0, F1_numero_consumption_industry, HC_i_0_clienti_iniziali) # suppongo sia il doppio dei clienti iniziali affinchè l' _i entrante non sia troppo svantaggiata
                    # valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_fallita_i_capital] = 2 * HC_i_0_clienti_iniziali # inizializzo
                    valore_HC_i_t_decimale_numero_clienti_di_ogni_i[azienda_fallita_i_capital] = HC_i_0_clienti_iniziali # inizializzo
                    #NW_i_t_liquid_assets_capital[azienda_fallita_i_capital] = NW_i_j_0_liquid_assets # inizializzo
                    # NW_i_t_liquid_assets_capital[azienda_fallita_i_capital] = NW_i_t_liquid_assets_capital[indice_estratto_capital_da_clonare]
                    NW_i_t_liquid_assets_capital[azienda_fallita_i_capital] = scala_lavoratori_capital_good_parametro_mail_roventini * NW_i_j_0_liquid_assets * w_salario_questo_turno
                    # Qua c'è un GROSSO problema! Se quando una _i muore copi una _i sopravvissuta in giro non potranno esserci più di NW_0 per ogni capital ovvero quanto avevano al t=0 siccome le _i non possono indebitarsi, possono assumere tanti lavoratori quanti sono al massimo gli NW_0
                    # ma l'aumento dei salari non è confinato da NW_0 anzi continua ad aumentare all'aumentare di AB quindi ad un certo punto NW_0 non basterà ad assumere nemmeno un lavoratore! Inoltre non può succedere nemmeno che le _i si rubino tra loro le NW_0_i assorbendo quelle delle sconfitte poichè producono solo quello richiesto dalle _j solo se possono assumere e assumono solo se possono pagare i lavoratori
                    for azienda_consumption_puntata in clienti_storici_con_cui_inizializzare_nuova_entrante:
                        # ma tra questi potrebbe essere uscito più volte l'indice della stessa azienda _j
                        while True:
                            if matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[azienda_fallita_i_capital, azienda_consumption_puntata] == 0:
                                matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[azienda_fallita_i_capital, azienda_consumption_puntata] = 1
                                brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_consumption_puntata] = np.append(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_consumption_puntata], azienda_fallita_i_capital)
                                clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_fallita_i_capital] = np.append(clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[azienda_fallita_i_capital], azienda_consumption_puntata)
                                # qua la dobbiamo aggiungere siccome prima avevamo sostituito l'intero np.array di _i fallita con uno vuoto e ora dobbiamo aggiungere le _j giuste che non sono doppioni prese dal vettore "clienti_storici_con_cui_inizializzare_nuova_entrante" dove ci sono indici di 2*HC_0 aziende _j
                                break
                            else:
                                azienda_consumption_puntata = np.random.randint(F1_numero_consumption_industry) # se c'è già un 1 in quel posto estraggo una nuova azienda consumption j che è puntata dalla stessa azienda di consumo i e ripeto il ciclo dove ora questa sarà la nuova
            else:
                # riferito a "if len(lista_estrazione_indici_i_non_fallite) > 0:"
                # Restart poichè tutte le capital _i sono fallite e quindi non è possibile prenderne nessuno come stampo da copiare
                restart_i_t_fallite_tutte = 1 # forse non serve, messo qua per similitudine con le consumption ma forse nel programma non verrà utilizzato
                contatore_restart_fallite_capital_storico_tutti_turni += 1
                #NW_i_t_liquid_assets_capital = NW_i_j_0_liquid_assets * np.ones(F2_numero_capital_industry)
                NW_i_t_liquid_assets_capital = scala_lavoratori_capital_good_parametro_mail_roventini * NW_i_j_0_liquid_assets * w_salario_questo_turno * np.ones(F2_numero_capital_industry)
                # Qua c'è un GROSSO problema! Se quando una _i muore copi una _i sopravvissuta in giro non potranno esserci più di NW_0 per ogni capital ovvero quanto avevano al t=0 siccome le _i non possono indebitarsi, possono assumere tanti lavoratori quanti sono al massimo gli NW_0
                # ma l'aumento dei salari non è confinato da NW_0 anzi continua ad aumentare all'aumentare di AB quindi ad un certo punto NW_0 non basterà ad assumere nemmeno un lavoratore! Inoltre non può succedere nemmeno che le _i si rubino tra loro le NW_0_i assorbendo quelle delle sconfitte poichè producono solo quello richiesto dalle _j solo se possono assumere e assumono solo se possono pagare i lavoratori
                A_average_labor_productivity_solo_capital_per_restart = np.sum(fake_innovazione_aziende_capital_good_A_i_tau)/len(fake_innovazione_aziende_capital_good_A_i_tau) # perchè se ad ogni turno muoiono tutte le _i, non innovano mai e la A e la B rimangono ferme!
                B_average_labor_productivity_solo_capital_per_restart = np.sum(fake_innovazione_aziende_capital_good_B_i_tau)/len(fake_innovazione_aziende_capital_good_B_i_tau)
                aziende_capital_good_A_i_tau = A_average_labor_productivity_solo_capital_per_restart * np.ones(F2_numero_capital_industry)
                aziende_capital_good_B_i_tau = B_average_labor_productivity_solo_capital_per_restart * np.ones(F2_numero_capital_industry)
                fake_innovazione_aziende_capital_good_A_i_tau = A_average_labor_productivity_solo_capital_per_restart * np.ones(F2_numero_capital_industry)
                fake_innovazione_aziende_capital_good_B_i_tau = B_average_labor_productivity_solo_capital_per_restart * np.ones(F2_numero_capital_industry)
                # clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità = [ np.random.randint(0, F1_numero_consumption_industry, 2 * HC_i_0_clienti_iniziali) for ci in range(F2_numero_capital_industry) ] # sono le righe della matrice F2xF1 solo che la stessa azienda di consumption j (cioè F1) potrebbe essere estratta più volte per cui servirà un loop per correggere questo
                clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità = [ np.random.randint(0, F1_numero_consumption_industry, HC_i_0_clienti_iniziali) for ci in range(F2_numero_capital_industry) ] # sono le righe della matrice F2xF1 solo che la stessa azienda di consumption j (cioè F1) potrebbe essere estratta più volte per cui servirà un loop per correggere questo
                # Però non è una lista "storico", semplicemente ad ogni turno aggiunge i nuovi clienti che sono in numero int( NC_i(t) ) quindi anche 0. La prossima lista di vettori è la versione speculare di questo, poi la matrice di adiacenza diretta riunisce entrambe le concezioni.
                # Imposto i nuovi clienti a "2*HC_i_0_clienti_iniziali" anzichè solo "HC_i_0_clienti_iniziali" come all'inizio perchè così ci sono meno possibilità che una _j non abbia capital _i da cui ordinare: non le siano arrivate brochure
                brochure_arrivate_alle_consumption_firms_con_indici_capital_firms = [ np.array([],int) for ci in range(F1_numero_consumption_industry) ] # sono F1 np.array che contengono gli indici _i delle capital firms alle quali le _j consumption firms mandano le brochure, NON sono i clienti storici HC_i(t). Sono le colonne della matrice.
                matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe = np.zeros((F2_numero_capital_industry, F1_numero_consumption_industry), dtype=int) # praticamente è una matrice di adiacenza di un grafo diretto
                for righe_matrice_o_F2 in range(len(clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità)):
                    # in realtà è una complicazione, bastava fare "for righe_matrice_o_F2 in range(F2_numero_capital_industry):" poichè quello è il numero di np.array nella lista
                    contatore_confronto = 0 # serve per dopo per tenere conto della posizione degli elementi del for su ogni np.array della lista poichè sarà necessario aggiornare l'indice se è già stato estratto 
                    for azienda_consumption_puntata in clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[righe_matrice_o_F2]:
                        confronto = azienda_consumption_puntata
                        while True:
                            if matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[righe_matrice_o_F2, azienda_consumption_puntata] == 0:
                                matrice_F2xF1_capital_puntano_a_consumption_clienti_storici_come_righe[righe_matrice_o_F2, azienda_consumption_puntata] = 1
                                brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_consumption_puntata] = np.append(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_consumption_puntata], righe_matrice_o_F2)
                                # così formo la lista (prima vuota) delle capital che hanno mandato una brochure alla consumption (la posizione dell'array nella lista) aggiungendo ogni volta una capital tra le F2 ad un np.array diverso delle consumption F1. Alla fine sono le colonne della matrice F2xF1 senza zeri però.
                                if confronto != azienda_consumption_puntata:
                                    clienti_storici_HC_i_sono_j_a_cui_i_manda_brochure_pubblicità[righe_matrice_o_F2][contatore_confronto] = azienda_consumption_puntata # serve poichè l'azienda potrebbe essere un duplicato nella estrazione dell'array sopra e quindi bisogna aggiornare l'array con l'azienda non duplicato
                                contatore_confronto = 1 + contatore_confronto # serve a tenere traccia della posizione azienda_consumption_puntata all'interno dell'np.array per poterlo poi cambiare
                                break
                            else:
                                azienda_consumption_puntata = np.random.randint(F1_numero_consumption_industry) # se c'è già un 1 in quel posto estraggo una nuova azienda consumption j che è puntata dalla stessa azienda di consumo i e ripeto il ciclo dove ora questa sarà la nuova
                # valore_HC_i_t_decimale_numero_clienti_di_ogni_i = 2 * HC_i_0_clienti_iniziali * np.ones(F2_numero_capital_industry) # serve poichè dobbiamo aumentarlo ogni turno aggiungendo al valore dell'azienda _i il NC_i(t)
                valore_HC_i_t_decimale_numero_clienti_di_ogni_i = 1 * HC_i_0_clienti_iniziali * np.ones(F2_numero_capital_industry) # serve poichè dobbiamo aumentarlo ogni turno aggiungendo al valore dell'azienda _i il NC_i(t)


    # Calcolo gli aggregati che andranno graficati a fine turno. Secondo me nei paper non viene considerato tanto il valore nominale di una grandezza ad esempio Y(t) quanto il suo tasso di crescita g(t)=( Y(t)-Y(t-1) )/Y(t-1)  quindi quello del turno attuale cioè (t) dipende da quello del turno passato cioè (t-1)
    # Poi comunque ne fa il LOGARITMO nei grafici!
    # Nel 2010 dice che PIL rispetta le "standard national account identities": Y(t) = C(t) + I(t) + DN(t) = Sum_{j=1}^{F1} Q_j(t) + Sum_{i=1}^{F2} Q_i(t)
    # Nota che si sta parlando della variazione degli inventories N(t) perchè nel PIL di questo turno Y(t) devono rientrare non tutte le quantità N_j che sono lì dal passato, ma solo quelle prodotte in questo turno.
    # E se queste fossero DN(t)<0 poichè si sono consumati i magazzini, secondo me, nella eq, dovrebbe essere DN(t)=0 poichè non si è prodotto oppure,
    # anzi NO, devono essere negative per cui vanno a ridurre le altre voci C e I: in partiolare, deve ridurre i consumi C poichè quelle merci consumate dalle scorte non sono state prodotte e quindi non sono nelle Q_j(t) attuali !
    D_N_j_t_variazione_magazzini_puo_essere_negativo = N_j_copiato_per_recuperare_merce_invenduta_prima_che_aziende_j_falliscano - N_j_copiato_per_calcolare_variazione_turno_precedente_magazzini # è un np.array questo # praticamente questo, al di là dei nomi che confusionari, sarebbe semplicemente N_fine_turno - N_inizio_turno
    D_N_t_variazione_magazzini_quantita = np.sum(D_N_j_t_variazione_magazzini_puo_essere_negativo)
    D_N_t_variazione_magazzini_quantita_x_prezzi = np.sum(D_N_j_t_variazione_magazzini_puo_essere_negativo * prezzi_tutte_consumption_firms_p_j_t)
    D_N_variazione_magazzini_quantita_storico_tutti_turni = np.append(D_N_variazione_magazzini_quantita_storico_tutti_turni, D_N_t_variazione_magazzini_quantita)
    D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, D_N_t_variazione_magazzini_quantita_x_prezzi)
    # sono degli array e il risultato della divisione è un array, non hai usato l'array "D_j_t_domanda_questo_turno_quantita" poichè in un caso viene usato la Q_j(t) se _j non ha prodotto abbastanza
    C_t_consumi_quantita = np.sum( S_j_t_attuale_total_sales_soldi/prezzi_tutte_consumption_firms_p_j_t )
    C_t_consumi_quantita_x_prezzi = np.sum(S_j_t_attuale_total_sales_soldi)
    C_consumi_quantita_storico_tutti_turni = np.append(C_consumi_quantita_storico_tutti_turni, C_t_consumi_quantita)
    C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, C_t_consumi_quantita_x_prezzi)
    # Gli Investimenti I(t) invece li avevi già calcolati prima nel loop delle _i per le macchine A comprate dalle _j che riuscivano a produrre oppure no e sono: I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno]  e  I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno]
    Y_t_pil_quantita = C_t_consumi_quantita + I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] + D_N_t_variazione_magazzini_quantita
    Y_t_pil_soldi_quantita_x_prezzi = C_t_consumi_quantita_x_prezzi + I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] + D_N_t_variazione_magazzini_quantita_x_prezzi
    Y_pil_quantita_storico_tutti_turni = np.append(Y_pil_quantita_storico_tutti_turni, Y_t_pil_quantita)
    Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni, Y_t_pil_soldi_quantita_x_prezzi)
    # Controllo come dice nel 2010 che il Pil sia uguale alla somma delle quantità Q_j(t)+Q_i(t) su tutte le industrie e ai loro valori monetari
    Y_t_controllo_pil_quantita = np.sum(Q_j_t_quantita_prodotta_nel_turno_attuale) + RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno]
    Y_t_controllo_pil_soldi_quantita_x_prezzi = np.sum(Q_j_t_quantita_prodotta_nel_turno_attuale * prezzi_tutte_consumption_firms_p_j_t) + RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno]
    print('--------')
    print('Nel turno', turno)
    #for azienda_j_consumption in range(F1_numero_consumption_industry):
        #if len(brochure_arrivate_alle_consumption_firms_con_indici_capital_firms[azienda_j_consumption]) == 0:
            #print('l\'azienda',azienda_j_consumption,'non ha brochure')
    print('Il debito totale di tutte le consumption è:', np.sum(Deb_j_turno_precedente_debito_consumption))
    print('Le aziende consumption fallite in questo turno sono:', quante_fallite_j_aziende_questo_turno_storico_tutti_turni[turno])
    print('Le aziende capital fallite in questo turno sono:', quante_fallite_i_aziende_questo_turno_storico_tutti_turni[turno])
    # print('La somma delle quantità RI+EI è uguale alla quantità con cui si calcolano gli I?', I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] == RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno])
    # if I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] != RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno]:
        # print('La somma delle quantità RI+EI è', RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno],'invece la quantità dei I è', I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno], 'e la differenza è',I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] - (RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno]) )
    # else:
        # print('e questa quantità I=RI+EI è', RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno])
    # print('In soldi RI+EI e I sono uguali?', I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] == RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno])
    # if I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] == RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno]:
        # print('e in soldi I=RI+EI valgono', RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno])
    # else:
        # print('In soldi RI+EI valgono', RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno], 'invece I vale', I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno], 'e la differenza è', I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] - (RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] + EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno]))
        # print('in particolare RI vale',RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno], 'e EI vale', EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno])
    # print('E\' vero che la quantità dell\'Output o Pil Y(t) è uguale calcolato nei due modi Y(t) = C(t) + I(t) + DN(t) == Sum_{j=1}^{F1}Q_j(t) + Sum_{i=1}^{F2}Q_i(t) ?', Y_t_controllo_pil_quantita == Y_t_pil_quantita)
    # if Y_t_controllo_pil_quantita != Y_t_pil_quantita:
        # print('...allora la differenza delle quantità è:', Y_t_controllo_pil_quantita - Y_t_pil_quantita)
    # print('E\' vero che il valore $ della quantità dell\'Output o Pil Y(t) è uguale calcolato nei due modi?', Y_t_controllo_pil_soldi_quantita_x_prezzi == Y_t_pil_soldi_quantita_x_prezzi)
    # if Y_t_controllo_pil_soldi_quantita_x_prezzi != Y_t_pil_soldi_quantita_x_prezzi:
        # print('...allora la differenza di valore $ è:', Y_t_controllo_pil_soldi_quantita_x_prezzi - Y_t_pil_soldi_quantita_x_prezzi)
        # print('Il valore $ del Pil Y(t) calcolato come somma Sum_{j=1}^{F1}Q_j(t) + Sum_{i=1}^{F2}Q_i(t) é:', Y_t_controllo_pil_soldi_quantita_x_prezzi)
        # print('Il valore $ del Pil Y(t) calcolato come Y(t) = C(t) + I(t) + DN(t) é:', Y_t_pil_soldi_quantita_x_prezzi)

    Emp_occupazione_storico_tutti_turni = np.append(Emp_occupazione_storico_tutti_turni, Emp_t_lavoratori_occupati)
    occupati_nelle_consumption_storico_tutti_turni = np.append(occupati_nelle_consumption_storico_tutti_turni, occupati_nelle_consumption)
    occupati_nelle_capital_storico_tutti_turni = np.append(occupati_nelle_capital_storico_tutti_turni, occupati_nelle_capital)
    numero_macchinari_K_intera_economia_storico_tutti_turni = np.append(numero_macchinari_K_intera_economia_storico_tutti_turni, np.sum(len(aziende_consumption_good[falso]) for falso in range(len(aziende_consumption_good))) )
    # Gli Investimenti RI(t) e EI(t) invece li avevi già calcolati prima nel loop delle _i per le macchine A comprate dalle _j che riuscivano a produrre oppure no e sono:
    # RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni  e  RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni
    # EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni  e  EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni
    if turno > 0:
        D_N_variazione_percentuale_magazzini_quantita_storico_tutti_turni = np.append(D_N_variazione_percentuale_magazzini_quantita_storico_tutti_turni, (D_N_variazione_magazzini_quantita_storico_tutti_turni[turno] - D_N_variazione_magazzini_quantita_storico_tutti_turni[turno-1])/D_N_variazione_magazzini_quantita_storico_tutti_turni[turno-1])
        D_N_variazione_percentuale_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(D_N_variazione_percentuale_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, (D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[turno] - D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1])/D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1])
        C_variazione_percentuale_consumi_quantita_storico_tutti_turni = np.append(C_variazione_percentuale_consumi_quantita_storico_tutti_turni, (C_consumi_quantita_storico_tutti_turni[turno] - C_consumi_quantita_storico_tutti_turni[turno-1])/C_consumi_quantita_storico_tutti_turni[turno-1])
        C_variazione_percentuale_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(C_variazione_percentuale_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, (C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[turno] - C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1])/C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1])
        I_variazione_percentuale_investimenti_quantita_storico_tutti_turni = np.append(I_variazione_percentuale_investimenti_quantita_storico_tutti_turni, (I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] - I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno-1])/I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno-1])
        I_variazione_percentuale_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(I_variazione_percentuale_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, (I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] - I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno-1])/I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno-1])
        Y_variazione_percentuale_pil_quantita_storico_tutti_turni = np.append(Y_variazione_percentuale_pil_quantita_storico_tutti_turni, (Y_pil_quantita_storico_tutti_turni[turno] - Y_pil_quantita_storico_tutti_turni[turno-1])/Y_pil_quantita_storico_tutti_turni[turno-1])
        Y_variazione_percentuale_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(Y_variazione_percentuale_pil_soldi_quantita_x_prezzi_storico_tutti_turni, (Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni[turno] - Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1])/Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1])
        Emp_variazione_percentuale_occupazione_storico_tutti_turni = np.append(Emp_variazione_percentuale_occupazione_storico_tutti_turni, (Emp_occupazione_storico_tutti_turni[turno] - Emp_occupazione_storico_tutti_turni[turno-1])/Emp_occupazione_storico_tutti_turni[turno-1])
        RI_variazione_percentuale_replacement_quantita_storico_tutti_turni = np.append(RI_variazione_percentuale_replacement_quantita_storico_tutti_turni, (RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] - RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno-1])/RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno-1])
        RI_variazione_percentuale_replacement_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(RI_variazione_percentuale_replacement_soldi_quantita_x_prezzi_storico_tutti_turni, (RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] - RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno-1])/RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno-1])
        EI_variazione_percentuale_expansion_quantita_storico_tutti_turni = np.append(EI_variazione_percentuale_expansion_quantita_storico_tutti_turni, (EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno] - EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno-1])/EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno-1])
        EI_variazione_percentuale_expansion_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(EI_variazione_percentuale_expansion_soldi_quantita_x_prezzi_storico_tutti_turni, (EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno] - EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno-1])/EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno-1])
        
        D_N_differenza_magazzini_quantita_storico_tutti_turni = np.append(D_N_differenza_magazzini_quantita_storico_tutti_turni, (D_N_variazione_magazzini_quantita_storico_tutti_turni[turno] - D_N_variazione_magazzini_quantita_storico_tutti_turni[turno-1]))
        D_N_differenza_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(D_N_differenza_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, (D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[turno] - D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1]))
        C_differenza_consumi_quantita_storico_tutti_turni = np.append(C_differenza_consumi_quantita_storico_tutti_turni, (C_consumi_quantita_storico_tutti_turni[turno] - C_consumi_quantita_storico_tutti_turni[turno-1]))
        C_differenza_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(C_differenza_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, (C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[turno] - C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1]))
        I_differenza_investimenti_quantita_storico_tutti_turni = np.append(I_differenza_investimenti_quantita_storico_tutti_turni, (I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno] - I_t_quantita_investment_consumption_firms_storico_tutti_turni[turno-1]))
        I_differenza_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(I_differenza_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, (I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno] - I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[turno-1]))
        Y_differenza_pil_quantita_storico_tutti_turni = np.append(Y_differenza_pil_quantita_storico_tutti_turni, (Y_pil_quantita_storico_tutti_turni[turno] - Y_pil_quantita_storico_tutti_turni[turno-1]))
        Y_differenza_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(Y_differenza_pil_soldi_quantita_x_prezzi_storico_tutti_turni, (Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni[turno] - Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni[turno-1]))
        Emp_differenza_occupazione_storico_tutti_turni = np.append(Emp_differenza_occupazione_storico_tutti_turni, (Emp_occupazione_storico_tutti_turni[turno] - Emp_occupazione_storico_tutti_turni[turno-1]))
        RI_differenza_replacement_quantita_storico_tutti_turni = np.append(RI_differenza_replacement_quantita_storico_tutti_turni, (RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno] - RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni[turno-1]))
        RI_differenza_replacement_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(RI_differenza_replacement_soldi_quantita_x_prezzi_storico_tutti_turni, (RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno] - RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[turno-1]))
        EI_differenza_expansion_quantita_storico_tutti_turni = np.append(EI_differenza_expansion_quantita_storico_tutti_turni, (EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno] - EI_t_quantita_expansion_investment_consumption_firms_storico_tutti_turni[turno-1]))
        EI_differenza_expansion_soldi_quantita_x_prezzi_storico_tutti_turni = np.append(EI_differenza_expansion_soldi_quantita_x_prezzi_storico_tutti_turni, (EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno] - EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[turno-1]))

    aziende_capital_good_A_i_tau = np.copy(fake_innovazione_aziende_capital_good_A_i_tau) # copio i vettori con le tecnologie nuove cioè con l'innovazione che saranno quelle vendute dalle capital alle consumption nel prossimo turno
    aziende_capital_good_B_i_tau = np.copy(fake_innovazione_aziende_capital_good_B_i_tau) # ho dovuto usare questi vettori "fake" perchè non potevo aggiornare i vettori originali nel loop delle capital _i, poichè mi servivano con le vecchie tecnologie durante il turno


print('----------------------------------------------------')
print('----------------------------------------------------')
print('SIMULAZIONE TERMINATA con successo!')
print('Restart delle consumption è avvenuto ', contatore_restart_fallite_consumption_storico_tutti_turni, 'volte, invece il restart delle capital è avvenuto', contatore_restart_fallite_capital_storico_tutti_turni, 'volte')
ora_di_fine = time.time()
# print("--- %s seconds ---" % (ora_di_fine - ora_dinizio) )
print("--- %s minutes ---" % np.int(np.round(((ora_di_fine - ora_dinizio))/60) )  )
print("--- %s hours and %s minutes ---" % ( np.int(((ora_di_fine - ora_dinizio))/(60*60) ) , np.int(np.round(60 * ( (((ora_di_fine - ora_dinizio))/(60*60) ) - np.int(((ora_di_fine - ora_dinizio))/(60*60) ) ) ) )  )   )
#%%
# FILTRO. PREPARAZIONE PER GRAFICI E STATISTICA
Y_senza_zeri_pil_soldi_quantita_x_prezzi_storico_tutti_turni = np.copy(Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni)
C_senza_zeri_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = np.copy(C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni)
I_t_senza_zeri_valore_monetario_investment_consumption_firms_storico_tutti_turni = np.copy(I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni)
EI_t_senza_zeri_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni = np.copy(EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni)
RI_t_senza_zeri_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni = np.copy(RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni)
# il problema è che np.log(0)=-infinito e siccome non puoi fare il log di zero li metto quasi a zero
for cip in range( len(EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni) ):
    if EI_t_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[cip] < 0.1:
        EI_t_senza_zeri_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni[cip] = 0.1/2 # 0.01/2
    if RI_t_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[cip] < 0.1:
        RI_t_senza_zeri_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni[cip] = 0.1/2 # 0.01/2
    if I_t_valore_monetario_investment_consumption_firms_storico_tutti_turni[cip] < 0.1:
        I_t_senza_zeri_valore_monetario_investment_consumption_firms_storico_tutti_turni[cip] = 0.1
    if C_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[cip] < 0.1:
        C_senza_zeri_consumi_soldi_quantita_x_prezzi_storico_tutti_turni[cip] = 0.1
    if Y_pil_soldi_quantita_x_prezzi_storico_tutti_turni[cip] < 0.1:
        Y_senza_zeri_pil_soldi_quantita_x_prezzi_storico_tutti_turni[cip] = 0.1


Y_log_senza_zeri_pil_soldi_quantita_x_prezzi_tutti_turni = np.log(Y_senza_zeri_pil_soldi_quantita_x_prezzi_storico_tutti_turni)
C_log_senza_zeri_consumi_soldi_quantita_x_prezzi_tutti_turni = np.log(C_senza_zeri_consumi_soldi_quantita_x_prezzi_storico_tutti_turni)
I_log_senza_zeri_investimenti_valore_monetario_tutti_turni = np.log(I_t_senza_zeri_valore_monetario_investment_consumption_firms_storico_tutti_turni)

T_periodo_piccolo = 6
T_periodo_grande = 32
K_filtro = 12
t_time_turno = [t for t in range(periodi_durata_simulazione_turni)]
t_time_turno_filtro = [t for t in range(K_filtro, len(t_time_turno)-K_filtro)]
Y_filtro_pil_soldi_quantita_x_prezzi_storico_tutti_turni = sm.tsa.filters.bkfilter(Y_log_senza_zeri_pil_soldi_quantita_x_prezzi_tutti_turni, low=T_periodo_piccolo, high=T_periodo_grande, K=K_filtro)
C_filtro_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = sm.tsa.filters.bkfilter(C_log_senza_zeri_consumi_soldi_quantita_x_prezzi_tutti_turni, low=T_periodo_piccolo, high=T_periodo_grande, K=K_filtro)
I_filtro_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni = sm.tsa.filters.bkfilter(I_log_senza_zeri_investimenti_valore_monetario_tutti_turni, low=T_periodo_piccolo, high=T_periodo_grande, K=K_filtro)
Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni = filter_baxter_king_1999(Y_log_senza_zeri_pil_soldi_quantita_x_prezzi_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni = filter_baxter_king_1999(C_log_senza_zeri_consumi_soldi_quantita_x_prezzi_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni = filter_baxter_king_1999(I_log_senza_zeri_investimenti_valore_monetario_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
#%%
# GRAFICI
font = {'family':'serif','color':'darkred','weight':'normal','size': 10.7,}
t_time_turno_differenze = [t for t in range(1, periodi_durata_simulazione_turni)] # per i grafici delle differenze

# grafico con doppia scala
scalacolore = 'deepskyblue'
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax1.plot(np.nan, scalacolore,label='Av. Produc. $\overline{A}(t)$')  # Creo un falso grafico con np.nan solo per potere collocare la legenda nella prima, infatti è ax1
# ax1.plot(t_time_turno, Y_senza_zeri_pil_soldi_quantita_x_prezzi_storico_tutti_turni,'r',label='GDP $Y(t)$')
# ax1.plot(t_time_turno, C_senza_zeri_consumi_soldi_quantita_x_prezzi_storico_tutti_turni,'k',label='Cons. $C(t)$')
# ax2.plot(t_time_turno, np.log(AB_average_labor_productivity_tutta_economia),scalacolore)
# ax1.legend(loc='best',prop={'size': 6})
# plt.title('Solo PIL Y, Consumi C e Produttività Media',fontdict = font)
# ax1.set_xlabel('Time t')
# ax1.set_ylabel('NON SONO LOGARITMI')
# ax2.set_ylabel('$\overline{A}(t)$', color=scalacolore)  # we already handled the x-label with ax1
# ax2.set_ylabel('$log \overline{A}(t)$', color=scalacolore)  # we already handled the x-label with ax1
# plt.grid()
#ax1.set_xlim(0,300)
# ax2.tick_params(axis='y', labelcolor=scalacolore) # per avere anche la scala dello stesso colore dell'etichetta
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('K+S soldi quantitàxprezzi solo Pil, consumi e produttività NO log seguire turni 1.jpg', dpi=650, transparent=False)
# plt.show()


plt.plot(t_time_turno, np.log(AB_average_labor_productivity_tutta_economia),'k',label='Av. Produc. $\overline{A}(t)$')
plt.plot(t_time_turno, np.log(A_average_productivity_solo_tutte_consumption_macchine),':',color='deepskyblue',label='Av. Produc. cons. $\overline{A_{j}}(t)$')
plt.plot(t_time_turno, np.log(A_average_productivity_solo_tutte_capital),'--',color='orange',label='Av. Produc. cap. $\overline{A_{i}}(t)$')
plt.plot(t_time_turno, np.log(B_average_productivity_solo_tutte_capital),'-.',color='firebrick',label='Av. Produc. cap. $\overline{B_{i}}(t)$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
# plt.title('Average Productivity',fontdict = font)
plt.xlabel('Time t')
plt.ylabel('log')
# plt.ylabel('average labor productivity')
plt.grid()
plt.savefig('K+S produttività assieme e separate log seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()


# '--', '--',
plt.plot(t_time_turno, np.log(Y_senza_zeri_pil_soldi_quantita_x_prezzi_storico_tutti_turni),color='r',label='GDP $Y(t)$')
plt.plot(t_time_turno, np.log(EI_t_senza_zeri_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni),color='black',label='Exp. Inv. $EI(t)$')
plt.plot(t_time_turno, np.log(RI_t_senza_zeri_valore_monetario_replacement_investment_consumption_firms_storico_tutti_turni),color='deepskyblue',label='Rep. Inv. $RI(t)$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
# plt.title('I scomposti in EI e RI logaritmi',fontdict = font)
plt.xlabel('Time t')
plt.ylabel('log')
plt.grid()
plt.ylim(0,)
plt.savefig('K+S EI e RI, e Pil log seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()


#plt.plot(t_time_turno_differenze[:len(t_time_turno_differenze)-contatore_restart_fallite_consumption_storico_tutti_turni], E_average_competitiveness_consumption_storico_tutti_turni[:len(t_time_turno_differenze)-contatore_restart_fallite_consumption_storico_tutti_turni],'g',label='$\overline{E(t)}$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('E average competitiveness per capire quando si annulla',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('NON SONO LOGARITMI')
#plt.grid()
# plt.savefig('K+S E average sectorial competitiveness NO log seguire turni 1.jpg', dpi=650, transparent=False)
#plt.show()


# plt.plot(t_time_turno_differenze, Y_differenza_pil_soldi_quantita_x_prezzi_storico_tutti_turni,'r',label='GDP $\Delta Y(t)$')
# plt.plot(t_time_turno_differenze, I_differenza_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni,'b',label='Inv. $\Delta I(t)$')
# plt.plot(t_time_turno_differenze, C_differenza_consumi_soldi_quantita_x_prezzi_storico_tutti_turni,'k',label='Cons. $\Delta C(t)$')
# plt.plot(t_time_turno_differenze, RI_differenza_replacement_soldi_quantita_x_prezzi_storico_tutti_turni,label='Repl. $\Delta RI(t)$')
# plt.plot(t_time_turno_differenze, EI_differenza_expansion_soldi_quantita_x_prezzi_storico_tutti_turni,label='Exp. $\Delta EI(t)$')
# plt.plot(t_time_turno_differenze, D_N_differenza_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni,label='Invent. $\Delta N(t)$')
# plt.plot(t_time_turno_differenze, Emp_differenza_occupazione_storico_tutti_turni,label='Emp. $\Delta L(t)$')
# plt.legend(loc='lower right', prop={'size': 6})
# plt.title('Differenze non percentuali come detrend',fontdict = font)
# plt.xlabel('Time t')
# plt.ylabel('$\Delta$')
# plt.grid()
#plt.xlim(0,300)
# plt.savefig('K+S differenze detrend seguire turni 1.jpg', dpi=650, transparent=False)
# plt.show()


#plt.plot(t_time_turno, D_expect_j_o_domanda_turno_precedente_storico_tutti_turni,'g',label='$D_{exp}(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('Domanda attesa (quantità) per capire quando si annulla',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('NON SONO LOGARITMI')
#plt.grid()
# plt.savefig('K+S quantità Domanda Attesa NO log seguire turni 1.jpg', dpi=650, transparent=False)
#plt.show()


plt.plot(t_time_turno, quante_fallite_j_aziende_questo_turno_storico_tutti_turni/F1_numero_consumption_industry,'k',label='consumption $j(t)$')
plt.plot(t_time_turno, quante_fallite_i_aziende_questo_turno_storico_tutti_turni/F2_numero_capital_industry,'b',label='capital $i(t)$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
# plt.title('Fallimenti consumption e capital firms',fontdict = font)
plt.xlabel('Time t')
plt.ylabel('consumption capital firms %')
# plt.ylabel('$\omega, \lambda$')
plt.grid()
plt.savefig('K+S fallimenti consumption e capital firms NO log seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()        


plt.plot(t_time_turno, Emp_occupazione_storico_tutti_turni/L_0_labor_supply_popolazione_lavoratori,'b',label='Emp % $L(t)/L_0$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
#plt.title('',fontdict = font)
plt.xlabel('Time t')
plt.ylabel('Employees %')
# plt.ylabel('$\omega, \lambda$')
plt.grid()
# plt.savefig('K+S occupazione NO log seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()


#':k','--r' oppure ':r',
plt.plot(t_time_turno, Emp_occupazione_storico_tutti_turni/L_0_labor_supply_popolazione_lavoratori,'k',label='Emp % $L(t)/L_0$')
plt.plot(t_time_turno, occupati_nelle_capital_storico_tutti_turni/L_0_labor_supply_popolazione_lavoratori,':b',label='Emp cap % $L_i(t)/L_0$')
plt.plot(t_time_turno, occupati_nelle_consumption_storico_tutti_turni/L_0_labor_supply_popolazione_lavoratori,':r',label='Emp cons % $L_j(t)/L_0$')
plt.legend(loc='best',prop={'size': 6})
plt.xlabel('Time t')
plt.ylabel('Employees %')
plt.grid()
plt.savefig('K+S occupazione divisa industrie.jpg', dpi=650, transparent=False)
plt.show()


# ':r', '--b', '-.k', '-',
plt.plot(t_time_turno, np.log(Y_senza_zeri_pil_soldi_quantita_x_prezzi_storico_tutti_turni),':r',label='GDP $Y(t)$')
plt.plot(t_time_turno, np.log(I_t_senza_zeri_valore_monetario_investment_consumption_firms_storico_tutti_turni),'--b',label='Inv. $I(t)$')
plt.plot(t_time_turno, np.log(C_senza_zeri_consumi_soldi_quantita_x_prezzi_storico_tutti_turni),'-.k',label='Cons. $C(t)$')
# plt.plot(t_time_turno, np.log(D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni),'-',c='hotpink',label='Var. Magaz. $\Delta N(t)$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
#plt.title('',fontdict = font)
plt.xlabel('Time t')
plt.ylabel('log')
plt.grid()
plt.ylim(0,)
plt.savefig('K+S soldi quantitàxprezzi log seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()


#plt.plot(t_time_turno, np.log(RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni),'deepskyblue',label='Rep. Inv. $RI(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('RI QUANTITA logaritmi',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('log')
#plt.grid()
#plt.show()


#plt.plot(t_time_turno, RI_t_quantita_replacement_investment_consumption_firms_storico_tutti_turni,'deepskyblue',label='Rep. Inv. $RI(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('RI QUANTITA NO logaritmi',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('quantita\'')
#plt.grid()
#plt.show()


#plt.plot(t_time_turno, numero_macchinari_K_intera_economia_storico_tutti_turni,'-',c='red',label='macchinari tutte consumption')
#plt.legend(loc='best',prop={'size': 6})
#plt.xlabel('Time t')
#plt.grid()
#plt.show()


#plt.plot(t_time_turno, np.log(produzione_massima_tutte_consumption_storico_tutti_turni),'--r',label='prod max consumption')
#plt.plot(t_time_turno, np.log(produzione_effettiva_tutte_consumption_storico_tutti_turni),':b',label='prod eff consumption')
#plt.legend(loc='best',prop={'size': 6})
#plt.xlabel('Time t')
#plt.grid()
#plt.show()


#plt.plot(t_time_turno,  [ np.sum(mi_mark_up_tutte_consumption_firms_storico_tutti_turni[brutto]) for brutto in range(len(mi_mark_up_tutte_consumption_firms_storico_tutti_turni))],'-',c=scalacolore,label='Consump. mark-up $\mu (t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.xlabel('Time t')
#plt.grid()
#plt.show()

 
#plt.plot(t_time_turno, D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni,'-',c='hotpink',label='Var. Magaz. $\Delta N(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.xlabel('Time t')
#plt.ylabel('NON SONO log')
#plt.grid()
#plt.show()

#plt.plot(t_time_turno, D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni,'-',c='hotpink',label='Var. Magaz. $\Delta N(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('Negative variazione del magazzini',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('NON SONO log')
#plt.grid()
#plt.ylim(10,-10000)
#plt.show()

#plt.plot(t_time_turno, np.log(D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni),'-',c='hotpink',label='Var. Magaz. $\Delta N(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('log dei positivi variazione del magazzini',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('log')
#plt.grid()
#plt.show()

#plt.plot(t_time_turno, np.log( - D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni),'-',c='hotpink',label='Var. Magaz. $\Delta N(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.title('log dei negativi resi positivi variazione del magazzini',fontdict = font)
#plt.xlabel('Time t')
#plt.ylabel('log')
#plt.grid()
#plt.show()


#plt.plot(t_time_turno_differenze, Y_variazione_percentuale_pil_soldi_quantita_x_prezzi_storico_tutti_turni,'r',label='GDP % $\Delta Y(t)$')
#plt.plot(t_time_turno_differenze, I_variazione_percentuale_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni,'b',label='Inv. % $\Delta I(t)$')
#plt.plot(t_time_turno_differenze, C_variazione_percentuale_consumi_soldi_quantita_x_prezzi_storico_tutti_turni,'k',label='Cons. % $\Delta C(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.xlabel('Time t')
#plt.ylabel('$\Delta$ %')
#plt.grid()
#plt.ylim(-1.5,5)
# plt.savefig('K+S differenze percentuali seguire turni 1.jpg', dpi=650, transparent=False)
#plt.show()


#plt.plot(t_time_turno_filtro, Y_filtro_pil_soldi_quantita_x_prezzi_storico_tutti_turni,':r',label='GDP % $Y(t)$')
#plt.plot(t_time_turno_filtro, I_filtro_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni,'--b',label='Inv. % $I(t)$')
#plt.plot(t_time_turno_filtro, C_filtro_consumi_soldi_quantita_x_prezzi_storico_tutti_turni,'-.k',label='Cons. % $C(t)$')
#plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
#plt.xlabel('Time t')
#plt.ylabel('log')
#plt.grid()
# plt.savefig('K+S filtro baxter sm seguire turni 1.jpg', dpi=650, transparent=False)
#plt.show()

plt.plot(t_time_turno_filtro, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni,':r',label='GDP % $Y(t)$')
plt.plot(t_time_turno_filtro, I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni,'--b',label='Inv. % $I(t)$')
plt.plot(t_time_turno_filtro, C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni,'-.k',label='Cons. % $C(t)$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
# plt.title('Baxter-King Filtro', fontdict = font)
plt.xlabel('Time t')
plt.ylabel('log')
plt.grid()
# plt.ylim(-2.5,1)
plt.savefig('K+S filtro baxter mio seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()
#%%
# STATISTICA
AGR_pil_average_growth_rate = (Y_log_senza_zeri_pil_soldi_quantita_x_prezzi_tutti_turni[periodi_durata_simulazione_turni-1] - Y_log_senza_zeri_pil_soldi_quantita_x_prezzi_tutti_turni[0])/periodi_durata_simulazione_turni
AGR_consumi_average_growth_rate = (C_log_senza_zeri_consumi_soldi_quantita_x_prezzi_tutti_turni[periodi_durata_simulazione_turni-1] - C_log_senza_zeri_consumi_soldi_quantita_x_prezzi_tutti_turni[0])/periodi_durata_simulazione_turni
AGR_investimenti_average_growth_rate = (I_log_senza_zeri_investimenti_valore_monetario_tutti_turni[periodi_durata_simulazione_turni-1] - I_log_senza_zeri_investimenti_valore_monetario_tutti_turni[0])/periodi_durata_simulazione_turni

std_Y_filtro = np.std(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, ddof=0) # se metti ddof=1 è N-1 al denominatore invece che N
std_C_filtro = np.std(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, ddof=0)
std_I_filtro = np.std(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, ddof=0)
print('     ***     ')
print('--------')
print('GDP | Avg. growth rate:',AGR_pil_average_growth_rate,', Std. dev. (bpf 6,32,12):',std_Y_filtro,', Rel.std. dev. (GDP):',std_Y_filtro/std_Y_filtro)
print('--------')
print('Consumption | Avg. growth rate:',AGR_consumi_average_growth_rate,', Std. dev. (bpf 6,32,12):',std_C_filtro,', Rel.std. dev. (GDP):',std_C_filtro/std_Y_filtro)
print('--------')
print('Investment | Avg. growth rate:',AGR_investimenti_average_growth_rate,', Std. dev. (bpf 6,32,12):',std_I_filtro,', Rel.std. dev. (GDP):',std_I_filtro/std_Y_filtro)
print('--------')
print('     ***     ')

# GDP vs GDP
GDP_corr_coef_zero = np.corrcoef(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
Y_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_restit_2_serie_lagate_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
GDP_corr_coef_meno_uno = np.corrcoef(Y_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
Y_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_restit_2_serie_lagate_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
GDP_corr_coef_meno_due = np.corrcoef(Y_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
Y_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_restit_2_serie_lagate_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
GDP_corr_coef_meno_tre = np.corrcoef(Y_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
Y_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_restit_2_serie_lagate_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
GDP_corr_coef_meno_quattro = np.corrcoef(Y_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
Y_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_restit_2_serie_lead_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
GDP_corr_coef_piu_uno = np.corrcoef(Y_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
Y_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_restit_2_serie_lead_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
GDP_corr_coef_piu_due = np.corrcoef(Y_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
Y_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_restit_2_serie_lead_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
GDP_corr_coef_piu_tre = np.corrcoef(Y_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
Y_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_restit_2_serie_lead_stessa_len(Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
GDP_corr_coef_piu_quattro = np.corrcoef(Y_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('     ***     ')
print('--------')
print('Correlation Structure | GDP | t-4:',GDP_corr_coef_meno_quattro,', t-3:',GDP_corr_coef_meno_tre,', t-2:',GDP_corr_coef_meno_due,', t-1:',GDP_corr_coef_meno_uno,', t:',GDP_corr_coef_zero,', t+1:',GDP_corr_coef_piu_uno,', t+2:',GDP_corr_coef_piu_due,', t+3:',GDP_corr_coef_piu_tre,', t+4:',GDP_corr_coef_piu_quattro)
# Cons. vs GDP (che è lag e lead)
C_corr_coef_zero = np.corrcoef(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
C_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
C_corr_coef_meno_uno = np.corrcoef(C_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
C_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
C_corr_coef_meno_due = np.corrcoef(C_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
C_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
C_corr_coef_meno_tre = np.corrcoef(C_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
C_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
C_corr_coef_meno_quattro = np.corrcoef(C_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
C_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
C_corr_coef_piu_uno = np.corrcoef(C_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
C_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
C_corr_coef_piu_due = np.corrcoef(C_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
C_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
C_corr_coef_piu_tre = np.corrcoef(C_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
C_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(C_filtro_mio_consumi_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
C_corr_coef_piu_quattro = np.corrcoef(C_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('--------')
print('Correlation Structure | Consumption | t-4:',C_corr_coef_meno_quattro,', t-3:',C_corr_coef_meno_tre,', t-2:',C_corr_coef_meno_due,', t-1:',C_corr_coef_meno_uno,', t:',C_corr_coef_zero,', t+1:',C_corr_coef_piu_uno,', t+2:',C_corr_coef_piu_due,', t+3:',C_corr_coef_piu_tre,', t+4:',C_corr_coef_piu_quattro)
# Inv. vs GDP (che è lag e lead)
I_corr_coef_zero = np.corrcoef(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
I_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
I_corr_coef_meno_uno = np.corrcoef(I_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
I_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
I_corr_coef_meno_due = np.corrcoef(I_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
I_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
I_corr_coef_meno_tre = np.corrcoef(I_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
I_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
I_corr_coef_meno_quattro = np.corrcoef(I_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
I_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
I_corr_coef_piu_uno = np.corrcoef(I_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
I_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
I_corr_coef_piu_due = np.corrcoef(I_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
I_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
I_corr_coef_piu_tre = np.corrcoef(I_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
I_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(I_filtro_mio_investimenti_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
I_corr_coef_piu_quattro = np.corrcoef(I_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('--------')
print('Correlation Structure | Investment | t-4:',I_corr_coef_meno_quattro,', t-3:',I_corr_coef_meno_tre,', t-2:',I_corr_coef_meno_due,', t-1:',I_corr_coef_meno_uno,', t:',I_corr_coef_zero,', t+1:',I_corr_coef_piu_uno,', t+2:',I_corr_coef_piu_due,', t+3:',I_corr_coef_piu_tre,', t+4:',I_corr_coef_piu_quattro)
# Change in stocks vs GDP (che è lag e lead)
D_N_senza_zeri_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = np.copy(D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni)
# il problema è che np.log(0)=-infinito
for cip in range( len(D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni) ):
    if D_N_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[cip] < 0.1:
        D_N_senza_zeri_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni[cip] = 0.1
D_N_log_senza_zeri_variazione_magazzini_soldi_tutti_turni = np.log(D_N_senza_zeri_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni)
D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni = filter_baxter_king_1999(D_N_log_senza_zeri_variazione_magazzini_soldi_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
D_N_corr_coef_zero = np.corrcoef(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
D_N_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
D_N_corr_coef_meno_uno = np.corrcoef(D_N_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
D_N_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
D_N_corr_coef_meno_due = np.corrcoef(D_N_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
D_N_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
D_N_corr_coef_meno_tre = np.corrcoef(D_N_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
D_N_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
D_N_corr_coef_meno_quattro = np.corrcoef(D_N_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
D_N_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
D_N_corr_coef_piu_uno = np.corrcoef(D_N_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
D_N_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
D_N_corr_coef_piu_due = np.corrcoef(D_N_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
D_N_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
D_N_corr_coef_piu_tre = np.corrcoef(D_N_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
D_N_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(D_N_filtro_mio_variazione_magazzini_soldi_quantita_x_prezzi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
D_N_corr_coef_piu_quattro = np.corrcoef(D_N_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('--------')
print('Correlation Structure | Change in stocks | t-4:',D_N_corr_coef_meno_quattro,', t-3:',D_N_corr_coef_meno_tre,', t-2:',D_N_corr_coef_meno_due,', t-1:',D_N_corr_coef_meno_uno,', t:',D_N_corr_coef_zero,', t+1:',D_N_corr_coef_piu_uno,', t+2:',D_N_corr_coef_piu_due,', t+3:',D_N_corr_coef_piu_tre,', t+4:',D_N_corr_coef_piu_quattro)
# Net investment vs GDP (che è lag e lead)
# secondo me i "Net Investment" sono i EI perchè gli altri, gli RI, sostituiscono il capitale già presente o morto
EI_log_senza_zeri_expansion_investment_consumption_soldi_tutti_turni = np.log(EI_t_senza_zeri_valore_monetario_expansion_investment_consumption_firms_storico_tutti_turni)
EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni = filter_baxter_king_1999(EI_log_senza_zeri_expansion_investment_consumption_soldi_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
EI_corr_coef_zero = np.corrcoef(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
EI_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
EI_corr_coef_meno_uno = np.corrcoef(EI_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
EI_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
EI_corr_coef_meno_due = np.corrcoef(EI_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
EI_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
EI_corr_coef_meno_tre = np.corrcoef(EI_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
EI_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
EI_corr_coef_meno_quattro = np.corrcoef(EI_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
EI_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
EI_corr_coef_piu_uno = np.corrcoef(EI_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
EI_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
EI_corr_coef_piu_due = np.corrcoef(EI_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
EI_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
EI_corr_coef_piu_tre = np.corrcoef(EI_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
EI_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(EI_filtro_mio_expansion_investment_soldi_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
EI_corr_coef_piu_quattro = np.corrcoef(EI_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('--------')
print('Correlation Structure | Net investment | t-4:',EI_corr_coef_meno_quattro,', t-3:',EI_corr_coef_meno_tre,', t-2:',EI_corr_coef_meno_due,', t-1:',EI_corr_coef_meno_uno,', t:',EI_corr_coef_zero,', t+1:',EI_corr_coef_piu_uno,', t+2:',EI_corr_coef_piu_due,', t+3:',EI_corr_coef_piu_tre,', t+4:',EI_corr_coef_piu_quattro)
# Employment vs GDP (che è lag e lead)
Emp_filtro_mio_occupazione_storico_tutti_turni = filter_baxter_king_1999(Emp_occupazione_storico_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
Emp_corr_coef_zero = np.corrcoef(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
Emp_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
Emp_corr_coef_meno_uno = np.corrcoef(Emp_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
Emp_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
Emp_corr_coef_meno_due = np.corrcoef(Emp_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
Emp_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
Emp_corr_coef_meno_tre = np.corrcoef(Emp_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
Emp_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
Emp_corr_coef_meno_quattro = np.corrcoef(Emp_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
Emp_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
Emp_corr_coef_piu_uno = np.corrcoef(Emp_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
Emp_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
Emp_corr_coef_piu_due = np.corrcoef(Emp_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
Emp_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
Emp_corr_coef_piu_tre = np.corrcoef(Emp_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
Emp_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Emp_filtro_mio_occupazione_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
Emp_corr_coef_piu_quattro = np.corrcoef(Emp_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('--------')
print('Correlation Structure | Employment | t-4:',Emp_corr_coef_meno_quattro,', t-3:',Emp_corr_coef_meno_tre,', t-2:',Emp_corr_coef_meno_due,', t-1:',Emp_corr_coef_meno_uno,', t:',Emp_corr_coef_zero,', t+1:',Emp_corr_coef_piu_uno,', t+2:',Emp_corr_coef_piu_due,', t+3:',Emp_corr_coef_piu_tre,', t+4:',Emp_corr_coef_piu_quattro)
# Unemployment Rate vs GDP (che è lag e lead)
Unemp_rate_disoccupazione_tasso_storico_tutti_turni = 1 - ( Emp_occupazione_storico_tutti_turni/L_0_labor_supply_popolazione_lavoratori )
Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni = filter_baxter_king_1999(Unemp_rate_disoccupazione_tasso_storico_tutti_turni, T_periodo_piccolo, T_periodo_grande, K_filtro)
Unemp_rate_corr_coef_zero = np.corrcoef(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni)[0,1]
Unemp_rate_un_lag_parte_da_lag, Y_un_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
Unemp_rate_corr_coef_meno_uno = np.corrcoef(Unemp_rate_un_lag_parte_da_lag, Y_un_lag_parte_da_zero)[0,1]
Unemp_rate_due_lag_parte_da_lag, Y_due_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
Unemp_rate_corr_coef_meno_due = np.corrcoef(Unemp_rate_due_lag_parte_da_lag, Y_due_lag_parte_da_zero)[0,1]
Unemp_rate_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
Unemp_rate_corr_coef_meno_tre = np.corrcoef(Unemp_rate_tre_lag_parte_da_lag, Y_tre_lag_parte_da_zero)[0,1]
Unemp_rate_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero = lagatore_doppio_input_restit_2_serie_lagate_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
Unemp_rate_corr_coef_meno_quattro = np.corrcoef(Unemp_rate_quattro_lag_parte_da_lag, Y_quattro_lag_parte_da_zero)[0,1]
Unemp_rate_un_lead_parte_da_zero, Y_un_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 1)
Unemp_rate_corr_coef_piu_uno = np.corrcoef(Unemp_rate_un_lead_parte_da_zero, Y_un_lead_parte_da_lead)[0,1]
Unemp_rate_due_lead_parte_da_zero, Y_due_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 2)
Unemp_rate_corr_coef_piu_due = np.corrcoef(Unemp_rate_due_lead_parte_da_zero, Y_due_lead_parte_da_lead)[0,1]
Unemp_rate_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 3)
Unemp_rate_corr_coef_piu_tre = np.corrcoef(Unemp_rate_tre_lead_parte_da_zero, Y_tre_lead_parte_da_lead)[0,1]
Unemp_rate_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead = avantatore_doppio_input_restit_2_serie_lead_stessa_len(Unemp_rate_filtro_mio_disoccupazione_tasso_storico_tutti_turni, Y_filtro_mio_pil_soldi_quantita_x_prezzi_storico_tutti_turni, 4)
Unemp_rate_corr_coef_piu_quattro = np.corrcoef(Unemp_rate_quattro_lead_parte_da_zero, Y_quattro_lead_parte_da_lead)[0,1]
print('--------')
print('Correlation Structure | Unemployment rate | t-4:',Unemp_rate_corr_coef_meno_quattro,', t-3:',Unemp_rate_corr_coef_meno_tre,', t-2:',Unemp_rate_corr_coef_meno_due,', t-1:',Unemp_rate_corr_coef_meno_uno,', t:',Unemp_rate_corr_coef_zero,', t+1:',Unemp_rate_corr_coef_piu_uno,', t+2:',Unemp_rate_corr_coef_piu_due,', t+3:',Unemp_rate_corr_coef_piu_tre,', t+4:',Unemp_rate_corr_coef_piu_quattro)
print('--------')
print('     ***     ')
print('Fine.')
print('     ***     ')
#%%
# Costruisco gli array che contengono le f_j(t) di alcune consumption estraendole dalla lista di np.array che è il vettore con tutti gli storici
quali_consumption_f_j_scelte_per_grafico = np.array([2, 37, 158, 189]) # queste ono le j scelte
contatore_consumption_estrarre = 0
f_market_share_tutti_turni_per_ogni_j_in_ordine = [np.array([]) for _ in range( len(f_market_share_tutte_consumption_firms_storico_tutti_turni[0]) )] # inizializzo la lista di np.array, che poi sarebbero in numero F1
for tutte_j_turno_da_estrarre in f_market_share_tutte_consumption_firms_storico_tutti_turni:
    for consumption_estrarre in tutte_j_turno_da_estrarre:
        # estraggo la f_j(t) che è "consumption_estrarre"
        f_market_share_tutti_turni_per_ogni_j_in_ordine[contatore_consumption_estrarre] = np.append(f_market_share_tutti_turni_per_ogni_j_in_ordine[contatore_consumption_estrarre], consumption_estrarre)
        contatore_consumption_estrarre += 1
    # qua è finito un turno: ovvero tutte le consumption del turno t sono state estratte e assegnate ai loro np.array, ora passo al turno t+1
    contatore_consumption_estrarre = 0 # poichè ora si deve ricominciare dalla prima consumption, la j=0, passando però al turno successivo, cioè t+1
for consumption_scelta_da_graficare in quali_consumption_f_j_scelte_per_grafico:
    plt.plot(t_time_turno, f_market_share_tutti_turni_per_ogni_j_in_ordine[consumption_scelta_da_graficare], label='$f_{%i}(t)$' % consumption_scelta_da_graficare)
plt.plot(t_time_turno, 1/len(f_market_share_tutte_consumption_firms_storico_tutti_turni[0]) * np.ones( len(t_time_turno) ), '-.k', label = r'$\frac{1}{F_{1}}$')
plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc='lower right', prop={'size': 6})
plt.title('Market Share $f_{j}(t)$ di alcune consumption per capire quando si annulla',fontdict = font)
plt.xlabel('Time t')
plt.ylabel('%')
plt.grid()
#plt.xlim(0,60)
plt.savefig('K+S market share alcune consumption percentuale seguire turni 1.jpg', dpi=650, transparent=False)
plt.show()
#%%
