"""Document and verify the two-video handoff before zipping it."""
import sys,csv,json,shutil,hashlib,zipfile,os,math,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(sys.argv[1]);S=Path(__file__).parent;N='GABARIT_2_VIDEOS_V2_20261018';O=R/'03_RESULTATS'/N
prov=json.loads((O/'PROVENANCE.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def read(p):
    with p.open(newline='') as f:yield from csv.DictReader(f)
shutil.copy2(S/'lire_une_frame.py',O/'outils/lire_une_frame.py')
assert (O/'RENDU_DEUX_VIDEOS.json').exists()
errs=list(read(O/'ERREURS_SUR_FRAMES_REFERENCE.csv'));frames=list(read(O/'RESUME_ERREURS_PAR_FRAME.csv'))
ea=np.array([float(r['error_selected_px']) for r in errs]);er=np.array([float(r['error_raw_px']) for r in errs])
stats=dict(reference_frames=len(frames),matched_sections=len(ea),mean_image_mae_selected_px=float(np.mean([float(r['selected_mae_px']) for r in frames])),section_mae_selected_px=float(abs(ea).mean()),section_rmse_selected_px=float(np.sqrt(np.mean(ea**2))),section_signed_bias_selected_px=float(ea.mean()),section_signed_sd_selected_px=float(ea.std(ddof=1)),raw_mae_same_sections_px=float(abs(er).mean()),image_dimensions=[1920,1080],scope='Same normalized decoded frames, reviewed human corridor. Development only; descriptive residual distribution, not calibrated uncertainty on arbitrary frames.',per_frame_error_for_tsv_record_available=False)
dump(O/'ERREUR_REFERENCE_RESUME.json',stats)
checks=[]
for v in prov['videos']:
    d=O/v['folder'];ff=list(read(d/'FRAMES_VITESSE_VECTEUR.csv'));assert len(ff)==v['normalized_frames']
    assert [int(r['frame_index']) for r in ff]==list(range(len(ff)))
    assert len({r['video_id'] for r in ff})==1
    data={int(r['frame_index']):r for r in ff};point_counts=collections.Counter();seen=set();measured=0;max_mid=0
    for p in read(d/'AXE_MEDIAN.csv'):
        i=int(p['frame_index']);r=data[i]
        for k in ['video_id','source_frame_index','video_time_s','is_new_source_frame','train_speed_m_s','real_elapsed_s','speed_status','axis_status','observation_status']:assert p[k]==r[k],(i,k)
        if p['x_px']:
            key=(i,int(p['segment_id']),int(p['point_index']));assert key not in seen;seen.add(key)
            x,y,l,rr=map(float,[p['x_px'],p['y_px'],p['left_x_px'],p['right_x_px']]);w,h=v['dimensions']
            assert 0<=l<x<rr<w and 0<=y<h
            max_mid=max(max_mid,abs(x-(l+rr)/2));assert max_mid<1e-6
            assert abs(float(p['width_px'])-(rr-l))<1e-6
            point_counts[i]+=1
            seg=json.loads(r['axis_segments_json'])[int(p['segment_id'])];q=seg[int(p['point_index'])];assert abs(q[0]-x)<1e-6 and q[1]==y
            if p['error_abs_px']:
                assert p['reference_image_id'] and v['video_id']=='TSV_03676'
                assert abs(abs(float(p['error_signed_px']))-float(p['error_abs_px']))<1e-8
                assert abs(x-float(p['reference_x_px'])-float(p['error_signed_px']))<1e-8;measured+=1
        elif int(r['axis_points'])>0:raise AssertionError('Empty row despite valid vector')
    for i,r in data.items():
        assert point_counts[i]==int(r['axis_points'])
        assert abs(float(r['video_time_s'])-i/v['fps'])<1e-8
        if r['train_speed_m_s']:assert abs(float(r['train_speed_m_s'])-float(r['train_speed_km_h'])/3.6)<1e-8
        if v['video_id']=='TSV_03676':
            assert bool(r['train_speed_m_s']) == (i<=5370)
            if i<=5370:assert r['speed_status']=='estimated_interval_mean_from_burned_gps_unvalidated' and r['time_basis']=='interpolated_burned_clock_unvalidated'
            assert not r['utc_time'] and not r['speed_uncertainty_m_s']
    spec=json.loads((d/'SOURCE_ET_DECOUPAGE.json').read_text())
    image_rows=list(read(d/'IMAGES_COMMUNES/MANIFESTE_IMAGES.csv'))
    for r in image_rows:
        im=Image.open(d/'IMAGES_COMMUNES'/r['image']).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==r['rgb_sha256']==spec['shared_rgb_sha256'][r['frame_index']]
    checks.append(dict(video_id=v['video_id'],frames=len(ff),axis_points=sum(point_counts.values()),duplicate_point_keys=0,all_frame_vectors_match_point_csv=True,measured_error_points=measured,verified_shared_pngs=len(image_rows),max_midpoint_error_px=max_mid))
# Contact sheet: factual decoded images, no generated image content.
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
sheet=Image.new('RGB',(1440,900),'#f4f6f5');draw=ImageDraw.Draw(sheet)
for row,v in enumerate(prov['videos']):
    d=O/v['folder'];images=sorted((d/'IMAGES_COMMUNES').glob('*.png'))
    picks=[images[0],images[len(images)//2],images[-1]]
    for col,p in enumerate(picks):
        im=Image.open(p).convert('RGB');im.thumbnail((460,360));x=col*480+10;y=row*440+55;sheet.paste(im,(x,y))
        draw.text((x,row*440+15),p.stem,fill='#173944',font=font)
        draw.text((x,row*440+405),'Entrée originale commune · sans tracé',fill='#536c74',font=font)
sheet.save(O/'APERCU_IMAGES_COMMUNES.jpg',quality=90)
mae=stats['mean_image_mae_selected_px']
msg=f'''Salut ! Voici le paquet V2 de notre équipe pour comparer nos détections avec celles de l’autre équipe sur les MÊMES images.

1. TSV_record : 7 608 frames à 5 images/s, image originale 1280 × 720. Vitesse et GPS fournis quand disponibles ; 36 vitesses manquent. La vidéo est accélérée : utiliser real_elapsed_s pour le déplacement du train, pas video_time_s.
2. 03676_Track insp 9_SOS_0123_20170109ok.AVI : 5 399 frames à 30 images/s, environ 180 s, image 1920 × 1080. Vitesse estimée à partir de 19 positions/heures GPS visibles et relues dans la vidéo : moyenne sur des intervalles de 9 à 11 s, non validée, frames 0 à 5370 couvertes ; 28 dernières vides. Le statut distingue cette estimation de la télémétrie fournie pour TSV_record. Il y a 94 frames répétées pour conserver la durée, signalées dans le CSV.

Dans chaque dossier : FRAMES_VITESSE_VECTEUR.csv donne une ligne par frame avec la vitesse et le vecteur d’axe ; AXE_MEDIAN.csv détaille les points, les deux bords et les erreurs mesurées lorsqu’une annotation existe. IMAGES_COMMUNES contient les PNG exacts à donner à l’autre équipe pour commencer. Joindre les résultats sur video_id + frame_index, et comparer les points au même y_px. Toutes les coordonnées sont dans l’image entière d’origine.

Pour 03676, nous avons retrouvé les 12 références pixel par pixel et ajouté leurs erreurs sur les frames correspondantes (MAE par image moyenne {mae:.2f} px sur l’axe brut retenu, développement seulement). Ailleurs l’erreur est inconnue, donc vide. Les masques de référence sont fournis pour que l’autre équipe calcule ses erreurs sur exactement les mêmes images. Attention : l’ancien nom frame_000000450 correspond à la frame exportée 458, pas 450. Le script fourni reproduit notre découpage.
'''
(O/'MESSAGE_A_ENVOYER.txt').write_text(msg)
text=f'''PAQUET V2 — NOTRE ÉQUIPE → GABARIT ET AUTRE ÉQUIPE RAILS

Commencer par LIRE_MOI.html. Donner tout ce dossier aux membres concernés, pas des CSV pris dans différentes versions.

VIDÉOS
- 01_TSV_RECORD : vidéo source TSV_record/video.mp4. GPS/vitesse disponibles (avec lacunes). 7608 frames, 5 fps, 1280×720, 1521.6 secondes de lecture. Temps physique couvert : 15560.008 secondes entre première et dernière frame.
- 02_TSV_03676_180S : vidéo source 03676_Track insp 9_SOS_0123_20170109ok.AVI. 5399 frames normalisées, 30 fps, 1920×1080, 179.9667 secondes. 5305 images réellement décodées et 94 répétitions. Vitesse expérimentale calculée à partir des coordonnées GPS incrustées : 19 ancrages relus, 18 moyennes sur 9 à 11 secondes, 5371 frames couvertes et 28 vides. Pas de relevé instrumental indépendant. Ce n’est PAS la vidéo 03383.

CSV DE CHAQUE DOSSIER
FRAMES_VITESSE_VECTEUR.csv : UNE LIGNE PAR FRAME. axis_segments_json contient le vecteur sous la forme [[[x,y],[x,y],...], ...], une liste par morceau continu. La vitesse est train_speed_m_s (ou train_speed_km_h). Le temps vidéo sert à ouvrir la bonne image ; le temps réel, quand connu, sert à la trajectoire du train.
AXE_MEDIAN.csv : UNE LIGNE PAR POINT, avec x_px, y_px, left_x_px, right_x_px, width_px, tangente et erreur si mesurée. Une frame sans axe a une ligne avec coordonnées vides.
EXEMPLE_SEQUENCE_FRAMES.csv et EXEMPLE_SEQUENCE_POINTS.csv : petits extraits correspondant aux PNG d’IMAGES_COMMUNES.

ERREURS
- 03676 : les 12 références humaines sont appariées exactement aux pixels décodés. Les colonnes d’erreur sont calculées sur les mêmes frames et aux mêmes hauteurs. Moyenne des MAE par image = {mae:.4f} px ; autres statistiques dans ERREUR_REFERENCE_RESUME.json. Ce sont des références de développement et elles ont leur imprécision humaine.
- TSV_record et autres frames de 03676 : pas de référence annotée, donc erreur inconnue. Une valeur vide ne vaut pas zéro ; aucune précision en mètres n’est validée.
- Les 12 masques dans REFERENCES_12 utilisent background=0, couloir=1, ignore=255. Pour l’autre équipe : relier au PNG par le numéro canonique et calculer l’erreur au même y. Ne pas considérer ce jeu comme un test final après avoir réglé la fusion dessus.

PROTOCOLE POUR L’AUTRE ÉQUIPE
1. Commencer par les PNG IMAGES_COMMUNES. Ils sont les entrées originales sans axes ni masque dessiné. Ne pas utiliser VIDEO_ANNOTEE.mp4 comme entrée du réseau.
2. Retourner toutes les coordonnées dans l’image entière d’origine. L’origine est en haut à gauche ; x augmente à droite et y vers le bas. Il n’est pas nécessaire d’utiliser notre recadrage interne, mais il faut annuler son propre recadrage/redimensionnement avant l’export.
3. Indexer les images à partir de 0. La clé de frame est (video_id, frame_index). La clé de comparaison de points est (video_id, frame_index, y_px). segment_id est local à chaque modèle/frame et ne doit pas être supposé identique entre équipes.
4. Pour la vidéo complète, télécharger l’ORIGINAL indiqué dans SOURCE_ET_DECOUPAGE.json ; vérifier le SHA256. Le script outils/extraire_frames_communes.py normalise d’abord la cadence puis sélectionne les frames. Ne pas faire simplement cap.set(frame_450) avec un autre décodeur.
5. Utiliser is_new_source_frame et observation_status. held_duplicate signifie que le tracé précédent est maintenu pour l’affichage, pas une nouvelle observation indépendante. source_frame_index est le numéro avant normalisation ; frame_index est le numéro dans la vidéo fournie.
6. Ne jamais transférer le GPS de TSV_record à 03676. Sur 03676, train_speed_m_s est une ESTIMATION MOYENNE par intervalle, pas une vitesse instantanée : speed_status et speed_is_estimated le signalent. real_elapsed_s est interpolé depuis l’heure incrustée. Le fuseau et l’incertitude sont inconnus. Lire VITESSE_ESTIMEE_GPS/LIRE_MOI.txt et vérifier les ancrages avant usage. Les 28 dernières frames restent sans vitesse, sans extrapolation.

EXEMPLES DE COMMANDES (depuis ce dossier)
python3 -m pip install -r outils/requirements.txt
python3 outils/lire_une_frame.py --dossier 01_TSV_RECORD --frame 50
python3 outils/lire_une_frame.py --dossier 02_TSV_03676_180S --frame 1375
python3 outils/extraire_frames_communes.py --video /chemin/vers/video_originale.mp4 --spec 01_TSV_RECORD/SOURCE_ET_DECOUPAGE.json --out images_record_communes

Les chemins comportant des espaces doivent être entourés de guillemets. Par défaut l’extracteur sort le même petit lot que celui livré ; ajouter --start 0 --count 7608 pour TSV_record entier ou --start 0 --count 5399 pour 03676 entier. Une extraction complète en PNG occupe beaucoup plus d’espace que le MP4 ; elle n’est pas nécessaire pour le premier échange.

PROVENANCE
Un seul modèle retenu : DeepLabV3Plus ResNet50, préapprentissage public ego-path puis adaptation TSV, axe brut retenu pour les calculs ; lissage comparé mais non retenu pour cette livraison. Le lissage historique augmentait la MAE sur les 12 mêmes frames (environ 3,8 → 5,3 px). Les coordonnées brutes sont donc retenues ; les vidéos ont été retracées avec ces coordonnées. Les prédictions existantes sont réexportées ici ; aucun des nouveaux modèles pilotes n’est utilisé automatiquement. Le CSV V2 distingue maintenant les deux sources, les répétitions, les erreurs mesurées et les vitesses absentes.
'''
(O/'LIRE_MOI.txt').write_text(text)
css='body{font:17px/1.65 -apple-system,BlinkMacSystemFont,sans-serif;max-width:1050px;margin:0 auto;padding:45px 30px;background:#f4f7f6;color:#173944}h1{font-size:45px;line-height:1.1}h2{font-size:27px;margin-top:38px}a{color:#006c78}table{border-collapse:collapse;width:100%;background:white;font-size:15px}td,th{padding:13px;text-align:left;border-bottom:1px solid #dae4e0}th{background:#dfecea}.box{background:white;padding:26px;border-radius:12px;margin:25px 0}.important{background:#e1f2ec;border-left:5px solid #008b83;padding:20px}code{background:#e4ece9;padding:2px 4px}pre{white-space:pre-wrap;font-size:13px}img,video{width:100%;height:auto}summary{cursor:pointer;font-weight:bold}li{padding:5px}small{color:#5c7179}'
html=f'''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Deux vidéos — axe, vitesse et erreurs</title><style>{css}</style></head><body><p>TSV · paquet commun V2 · 18 octobre 2026</p><h1>Deux vidéos. Les mêmes frames. Des données comparables.</h1><p>Ce dossier contient <strong>notre sortie sélectionnée</strong>. L’autre équipe utilise les mêmes images pour fournir sa sortie ; le gabarit pourra les aligner et étudier leur fusion.</p><div class="important"><strong>À envoyer :</strong> ce dossier complet ou son ZIP. Lire <a href="MESSAGE_A_ENVOYER.txt">le message court à copier</a>. Les vidéos avec tracés servent à regarder ; les PNG communs servent d’entrée aux modèles.</div><h2>1. De quelles vidéos parle-t-on ?</h2><table><thead><tr><th>Source</th><th>Frames / durée / taille</th><th>Vitesse</th><th>Erreur connue ?</th></tr></thead><tbody><tr><td><a href="01_TSV_RECORD/SOURCE_ET_DECOUPAGE.json">TSV_record / video.mp4</a></td><td>7 608 · 25 min 21,6 s de lecture · 1280 × 720</td><td>Oui, sauf 36 frames ; temps réel et GPS fournis avec indicateurs qualité.</td><td>Pas d’annotation de référence pour ces frames : erreur vide.</td></tr><tr><td><a href="02_TSV_03676_180S/SOURCE_ET_DECOUPAGE.json">03676_Track insp 9…AVI</a></td><td>5 399 · 179,97 s · 1920 × 1080</td><td>Estimation GPS sur 5 371 frames ; moyenne sur 9–11 s, non validée. 28 dernières vides. Voir la méthode ci-dessous.</td><td>Oui sur les sections comparables des 12 frames annotées, vides ailleurs.</td></tr></tbody></table><p><strong>03676 n’est pas 03383.</strong> Nous avons retenu 03676 comme seconde vidéo parce qu’elle contient les 12 références humaines déjà disponibles.</p><h2>2. Quels CSV ouvrir ?</h2><div class="box"><p><strong>FRAMES_VITESSE_VECTEUR.csv</strong> : une ligne par frame ; vitesse et vecteur d’axe sous <code>axis_segments_json</code>.</p><p><strong>AXE_MEDIAN.csv</strong> : une ligne par point ; les limites, l’axe et les erreurs mesurées sont en colonnes.</p><p>Une frame correspond à plusieurs points. Plusieurs listes de points représentent des morceaux séparés : ne pas relier les trous. Les unités de l’axe sont les pixels de l’image originale, celles de la vitesse sont explicitement km/h et m/s.</p><table><tr><th>TSV_record</th><th>03676</th></tr><tr><td><a href="01_TSV_RECORD/FRAMES_VITESSE_VECTEUR.csv">CSV par frame</a><br><a href="01_TSV_RECORD/AXE_MEDIAN.csv">CSV des points</a><br><a href="01_TSV_RECORD/EXEMPLE_SEQUENCE_FRAMES.csv">Petit exemple pour commencer</a></td><td><a href="02_TSV_03676_180S/FRAMES_VITESSE_VECTEUR.csv">CSV par frame</a><br><a href="02_TSV_03676_180S/AXE_MEDIAN.csv">CSV des points</a><br><a href="02_TSV_03676_180S/EXEMPLE_SEQUENCE_FRAMES.csv">Petit exemple pour commencer</a></td></tr></table></div><h2>3. Comment avoir exactement les mêmes images ?</h2><ol><li>Commencer par les <strong>PNG livrés dans IMAGES_COMMUNES</strong> : 10 frames consécutives de TSV_record (50–59), et 30 frames consécutives de 03676 (1375–1404), plus ses autres frames de référence.</li><li>Garder le numéro dans le nom du PNG. Joindre sur <code>video_id + frame_index</code>, puis comparer les points au même <code>y_px</code>.</li><li>Le modèle de l’autre équipe peut redimensionner ou recadrer en interne ; ses résultats doivent revenir dans l’image entière, d’origine en haut à gauche.</li><li>Pour extraire davantage d’images, utiliser l’original et <a href="outils/extraire_frames_communes.py">le script commun</a> avec le fichier SOURCE_ET_DECOUPAGE.json. Il vérifie l’identité de la vidéo et applique la même normalisation.</li></ol><div class="important"><strong>Pourquoi les anciens noms ne suffisent pas :</strong> la vidéo 03676 a 5 305 images décodables ; la cadence régulière comporte 5 399 frames, dont 94 répétitions. L’ancienne image « frame 450 » est la frame normalisée <strong>458</strong> ; « frame 1350 » devient <strong>1375</strong>. La correspondance des 12 références a été vérifiée sur tous les pixels RGB.</div><img src="APERCU_IMAGES_COMMUNES.jpg" alt="Exemples des entrées originales communes"><h2>4. Comment utiliser les erreurs ?</h2><p>Sur les 12 frames de 03676, la moyenne des MAE par image sur l’axe brut retenu vaut <strong>{mae:.2f} pixels</strong>. L’erreur signée est <code>x_brut_prédit − x_annoté</code>, à la même hauteur y. Le détail est dans <a href="ERREURS_SUR_FRAMES_REFERENCE.csv">les erreurs par point</a> et <a href="RESUME_ERREURS_PAR_FRAME.csv">le résumé par frame</a>.</p><p>Les masques de référence sont joints dans <code>02_TSV_03676_180S/REFERENCES_12</code> : l’autre équipe peut évaluer ses prédictions sur ces mêmes images. Ce sont des références de développement déjà utilisées pour nos choix. La dispersion mesurée ne doit pas devenir une marge d’erreur garantie pour toutes les vidéos.</p><p>Pour toutes les autres frames, les champs d’erreur restent vides avec un statut explicite. Une confiance du réseau ou un désaccord entre les équipes ne remplace pas une erreur mesurée face à une annotation.</p><h2>5. Attention au temps du trajet</h2><p>Pour TSV_record, <strong>video_time_s</strong> localise la lecture, <strong>real_elapsed_s</strong> décrit le temps physique fourni : environ 15 560 s de trajet entre première et dernière frame contre 1 521,6 s de vidéo. Intégrer la vitesse sur le temps de lecture donnerait un avancement faux. Les lacunes et drapeaux GPS doivent être traités.</p><p>Pour 03676, une estimation est calculée à partir de 19 positions et heures GPS visibles, relues dans la vidéo : environ 15,1 à 39,1 km/h selon l’intervalle. Ce sont des <strong>vitesses moyennes sur 9 à 11 secondes, non validées indépendamment</strong>, affectées aux frames de l’intervalle ; aucune vitesse instantanée n’a été mesurée. Les colonnes speed_status et speed_is_estimated identifient ces valeurs. Le temps real_elapsed_s est interpolé entre les heures affichées ; UTC et incertitude restent inconnus. Les 28 dernières frames restent vides. <a href="02_TSV_03676_180S/VITESSE_ESTIMEE_GPS/LIRE_MOI.txt">Lire la méthode</a> et <a href="02_TSV_03676_180S/VITESSE_ESTIMEE_GPS/ANCRAGES_VISIBLES.jpg">vérifier les 19 ancrages visibles</a>. <a href="02_TSV_03676_180S/VITESSE_ESTIMEE_GPS/VITESSE_ESTIMEE.png">Voir le graphique de vitesse estimée</a>. Les frames répétées ont <code>is_new_source_frame=false</code> et <code>observation_status=held_duplicate</code> : ne pas les compter comme de nouvelles observations indépendantes.</p><h2>6. Voir les résultats</h2><details><summary>Vidéo TSV_record</summary><video controls preload="none" src="01_TSV_RECORD/VIDEO_ANNOTEE.mp4"></video></details><details><summary>Vidéo 03676 — environ 180 secondes</summary><video controls preload="none" src="02_TSV_03676_180S/VIDEO_ANNOTEE.mp4"></video></details><h2>7. Les fichiers pour programmer</h2><ul><li><a href="LIRE_MOI.txt">Instructions complètes et commandes</a></li><li><a href="COLONNES.json">Dictionnaire des colonnes</a></li><li><a href="outils/lire_une_frame.py">Petit lecteur Python d’une frame</a></li><li><a href="PROVENANCE.json">Sources, modèle et conventions</a></li><li><a href="VERIFICATION.json">Contrôles du paquet</a></li></ul><p><small>Aucun envoi externe effectué. Les pixels n’ont pas été convertis en mètres sans calibration ; le GPS localise le train et ne calibre pas automatiquement les limites des rails.</small></p></body></html>'''
(O/'LIRE_MOI.html').write_text(html)
dump(O/'VERIFICATION.json',dict(status='passed',checks=checks,reference_alignment_all_12_exact=True,new_training_used=False,metric_conversion=False,errors_only_on_reference_frames=True,other_team_outputs_present=False,limitations='Checks validate format, identities, algebraic consistency and reference matches. They do not certify route choice or physical accuracy.'))
for p in [S/'build_two_video_package.py',S/'finalize_two_video_package.py']:shutil.copy2(p,R/'scripts/fiabilisation_20261018'/p.name)
manifest={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='SHA256.json'};dump(O/'SHA256.json',manifest)
z=R/'02_A_PARTAGER'/(N+'.zip')
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
    for p in sorted(O.rglob('*')):
        if p.is_file():archive.write(p,str(Path(N)/p.relative_to(O)),compress_type=zipfile.ZIP_STORED if p.suffix.lower() in ['.mp4','.jpg','.png'] else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(z) as archive:
    assert archive.testzip() is None
    for name,digest in manifest.items():assert hashlib.sha256(archive.read(N+'/'+name)).hexdigest()==digest
print(json.dumps(dict(folder=str(O),zip=str(z),zip_bytes=z.stat().st_size,verification=checks,error_summary=stats),indent=2),flush=True)
