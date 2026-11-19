"""Grouped evidence, explicit abstention and final-capture separation."""
import math
from collections import defaultdict
import numpy as np


def aggregate_axis_errors(rows):
    """Each input row must carry image_id, capture_id and error_signed_px.

    Return equal-image and equal-capture summaries; never treat neighboring
    sections as replicate experiments. No automatic confidence interval.
    """
    images=defaultdict(list);capture_for={}
    for r in rows:
        image=str(r['image_id']);capture=str(r['capture_id']);error=float(r['error_signed_px'])
        if not image or not capture or not math.isfinite(error):raise ValueError('Identité de capture/image et erreur finie requises')
        if image in capture_for and capture_for[image]!=capture:raise ValueError('Même image dans plusieurs captures')
        capture_for[image]=capture;images[image].append(error)
    if not images:raise ValueError('Aucune référence mesurée')
    by_image=[];captures=defaultdict(list)
    for image,errors in sorted(images.items()):
        a=np.array(errors);mae=float(np.abs(a).mean());captures[capture_for[image]].append(mae)
        by_image.append(dict(image_id=image,capture_id=capture_for[image],sections=len(a),mae_px=mae,
                             rmse_px=float(np.sqrt(np.mean(a*a))),bias_px=float(a.mean())))
    image_values=[r['mae_px'] for r in by_image]
    result=dict(sections=sum(len(e) for e in images.values()),images=len(images),captures=len(captures),
       image_macro_mae_px=float(np.mean(image_values)),image_median_mae_px=float(np.median(image_values)),
       image_min_mae_px=min(image_values),image_max_mae_px=max(image_values),
       capture_macro_mae_px=float(np.mean([np.mean(v) for v in captures.values()])),
       confidence_interval=None,confidence_note='No interval computed: independence and sampling of captures must be established.',
       per_image=by_image)
    return result


def final_split_guard(registry, requested):
    """Fail closed on unknown identity, prior development use or shared trip."""
    if not requested:raise ValueError('Aucune nouvelle capture finale réservée')
    known={r['capture_id']:r for r in registry}
    if len(known)!=len(registry):raise ValueError('Identifiants de capture dupliqués')
    used=[r for r in registry if r.get('role') in ['train','development','benchmark']]
    used_groups={r.get('capture_group') for r in used};used_hashes={r.get('source_sha256') for r in used}
    selected=[]
    for ident in requested:
        if ident not in known:raise ValueError('Capture inconnue : '+ident)
        r=known[ident]
        if r.get('role')!='final_reserved' or r.get('identity_verified') is not True:
            raise ValueError('Capture non réservée ou provenance non vérifiée : '+ident)
        if not r.get('capture_group') or not r.get('source_sha256'):
            raise ValueError('Groupe de capture et empreinte requis')
        if r['capture_group'] in used_groups or r['source_sha256'] in used_hashes:
            raise ValueError('Capture/session déjà utilisée pour apprendre ou régler : '+ident)
        if r.get('predictions_seen_before_freeze') is not False:
            raise ValueError('Exposition antérieure aux prédictions non exclue : '+ident)
        selected.append(r)
    return dict(eligible=True,captures=len(selected),capture_groups=len({r['capture_group'] for r in selected}),
                note='Registry attestations still require verified provenance; hash separation alone is insufficient.')


def branch_metrics(reference, prediction):
    """Dedicated route choice metric. Reference branches need human evidence.

    schema: event_id, frame_index, branch_id; references have reviewed=True and
    route_known=True; predictions have status='accepted' or abstention.
    Missing predictions count as abstentions, never as correct route choices.
    """
    def index(rows):
        d={}
        for r in rows:
            k=(str(r['event_id']),int(r['frame_index']))
            if k in d:raise ValueError('Frame dupliquée dans un événement')
            d[k]=r
        return d
    ref,pred=index(reference),index(prediction);events=defaultdict(lambda:dict(known=0,accepted=0,correct=0,wrong=0,abstained=0))
    excluded=0
    for key,r in ref.items():
        if r.get('reviewed') is not True or r.get('route_known') is not True or not r.get('branch_id'):
            excluded+=1;continue
        e=events[key[0]];e['known']+=1;p=pred.get(key)
        if not p or p.get('status')!='accepted' or not p.get('branch_id'):e['abstained']+=1;continue
        e['accepted']+=1
        if p['branch_id']==r['branch_id']:e['correct']+=1
        else:e['wrong']+=1
    totals={k:sum(e[k] for e in events.values()) for k in ['known','accepted','correct','wrong','abstained']}
    return dict(**totals,reference_frames_excluded=excluded,events=len(events),
      coverage=totals['accepted']/totals['known'] if totals['known'] else None,
      correct_among_accepted=totals['correct']/totals['accepted'] if totals['accepted'] else None,
      success_all_known=totals['correct']/totals['known'] if totals['known'] else None,
      events_with_wrong_branch=sum(e['wrong']>0 for e in events.values()),
      event_macro_success=float(np.mean([e['correct']/e['known'] for e in events.values()])) if events else None,
      per_event=dict(events),score_status='measured_against_reviewed_route' if totals['known'] else 'reference_missing')


def profile_alerts(segments, height, width):
    """Triage only. A fragmented model mask is not proof of a rail switch."""
    points=[p for s in segments for p in s];flags=[]
    if not points:return ['axis_absent']
    if len(segments)>1:flags.append('axis_fragmented')
    if max(p['y'] for p in points)<.8*height:flags.append('near_vehicle_anchor_missing')
    ordered=sorted(segments,key=lambda s:min(p['y'] for p in s))
    for a,b in zip(ordered,ordered[1:]):
        p=max(a,key=lambda q:q['y']);q=min(b,key=lambda q:q['y'])
        if abs(p['center_x']-q['center_x'])>.04*width:
            flags.append('large_lateral_gap_between_segments');break
    return flags
