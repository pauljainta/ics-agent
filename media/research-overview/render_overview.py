#!/usr/bin/env python3
"""Render the original 150-second ICS-Agent research animation.

Install Pillow and imageio-ffmpeg, then run from this directory:
    python render_overview.py
Use --preview to render only the contact sheet and poster.
All visuals and the quiet synthesized soundtrack are original. No plant footage,
exploitation tools, third-party media, deployment addresses, or secrets are used.
"""
from __future__ import annotations
import argparse
from array import array
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import wave
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
if (ROOT / '_deps').exists():
    sys.path.insert(0, str(ROOT / '_deps'))
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None
W, H, FPS, DURATION = 1280, 720, 15, 150
BG = '#080e19'
PANEL = '#101c2c'
LINE = '#24374c'
WHITE = '#eff5ff'
MUTED = '#95abc3'
CYAN = '#5ee6e8'
ORANGE = '#ff986c'
RED = '#ff6577'
GREEN = '#a5efab'
PURPLE = '#bdadff'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
FONTS = {}

def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in FONTS:
        stem = 'DejaVuSansMono' if mono else 'DejaVuSans'
        filename = stem + ('-Bold' if bold else '') + '.ttf'
        choices = [ROOT / 'fonts' / filename, FONT_DIR / filename]
        chosen = next((p for p in choices if p.exists()), None)
        FONTS[key] = ImageFont.truetype(str(chosen), size) if chosen else ImageFont.load_default(size=size)
    return FONTS[key]

SCENES = [
    (0, 10, 'RESEARCH OVERVIEW', 'When software reaches the physical world',
     'Tool-using AI agents. Industrial control systems. Measurable consequences.', 'intro'),
    (10, 24, '01  ENTERPRISE IT', 'Start with an assumed foothold',
     'A conceptual enterprise endpoint is already compromised. The agent begins from a defined, authorized research position.', 'it'),
    (24, 38, '02  THE TRUST BOUNDARY', 'Crossing IT into operational technology',
     'The storyline follows a permitted path toward an engineering interface. Segmentation, identity, and action authority define the boundary.', 'boundary'),
    (38, 52, '03  OT RECONNAISSANCE', 'Build an operational picture',
     'Inspect allowed surfaces. Identify devices and process roles. Read observations. Connect digital evidence to pumps, valves, and tanks.', 'recon'),
    (52, 67, '04  TOOL-USING AGENT', 'Observe. Plan. Act. Verify.',
     'The agent selects structured actions. Validation checks the proposal. The runtime returns new observations and a traceable process response.', 'loop'),
    (67, 84, '05  MODELED PHYSICAL EFFECTS', 'A digital action can change a process',
     'Manipulated observations or control actions can alter modeled flow and tank levels. The evaluation tracks actual process state separately from displayed readings.', 'process'),
    (84, 97, 'HISTORICAL CONTEXT  /  2010', 'Stuxnet: industrial control was the target',
     'Stuxnet targeted Siemens ICS and manipulated centrifuge-control logic. Its lesson: trusted engineering paths can carry adversarial physical intent.', 'stuxnet'),
    (97, 110, 'HISTORICAL CONTEXT  /  2016', 'Industroyer: the language of the grid',
     'Industroyer used industrial protocols to interact with substation equipment. It was involved in the December 2016 Kyiv power outage.', 'industroyer'),
    (110, 121, 'HISTORICAL CONTEXT  /  2017', 'TRITON: the safety layer is in scope',
     'TRITON targeted Triconex safety controllers. Safety-system integrity must remain separate from an agent’s confidence or a successful tool call.', 'triton'),
    (121, 135, '06  DEFENDER RESPONSE', 'Contain the action. Recover the process.',
     'A defender observes anomalies and applies permitted containment. Compare immediate, delayed, and observation-only authority against safety and availability.', 'defense'),
    (135, 146, '07  EVIDENCE AND EVALUATION', 'Measure outcomes beyond the tool call',
     'Inspect trajectories, safety violations, response time, downtime, and recovery. Compare agent policies with scripted and rule-based baselines.', 'evaluation'),
    (146, 150, 'ONGOING RESEARCH', 'ICS-Agent',
     'AI systems engineering meets industrial cybersecurity.', 'outro'),
]

SOURCES = {
    'stuxnet': ('ENISA + Langner · Stuxnet analysis', 'https://www.enisa.europa.eu/news/enisa-news/stuxnet-analysis'),
    'industroyer': ('ESET Research · Industroyer retrospective', 'https://www.welivesecurity.com/2022/06/13/industroyer-cyber-weapon-brought-down-power-grid/'),
    'triton': ('NCSC · TRITON Malware Targeting Safety Controllers', 'https://www.ncsc.gov.uk/information/triton-malware-targeting-safety-controllers'),
}

def mix(a, b, p): return a + (b-a)*p

def ease(p):
    p = max(0, min(1, p))
    return p*p*(3-2*p)

def text(d, xy, value, size=22, color=WHITE, bold=False, mono=False, anchor=None):
    d.text(xy, value, font=font(size, bold, mono), fill=color, anchor=anchor)

def wrapped(d, value, x, y, max_width, size=23, color=MUTED, line_height=None, bold=False):
    line_height = line_height or int(size*1.45)
    out, line = [], ''
    for word in value.split():
        trial = (line+' '+word).strip()
        if d.textlength(trial, font=font(size, bold)) > max_width and line:
            out.append(line); line=word
        else: line=trial
    out.append(line)
    for line in out:
        text(d, (x,y), line, size, color, bold)
        y += line_height
    return y

def rr(d, box, fill=PANEL, outline=LINE, width=1, radius=16):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def pill(d, x, y, label, color=CYAN):
    width = d.textlength(label, font=font(14, True, True)) + 28
    rr(d, (x,y,x+width,y+29), fill=BG, outline=color, radius=14)
    text(d, (x+14,y+5), label, 14, color, True, True)
    return width

def packet(d, points, t, color=CYAN, speed=0.16, offset=0):
    lengths=[math.dist(a,b) for a,b in zip(points,points[1:])]
    distance=((t*speed+offset)%1)*sum(lengths)
    for i, length in enumerate(lengths):
        if distance<=length:
            a,b=points[i],points[i+1]; p=distance/length if length else 0
            x,y=mix(a[0],b[0],p),mix(a[1],b[1],p)
            d.ellipse((x-8,y-8,x+8,y+8),fill=color)
            d.ellipse((x-3,y-3,x+3,y+3),fill=WHITE)
            break
        distance-=length

def server(d,x,y,color=CYAN):
    for i in range(3):
        rr(d,(x,y+i*19,x+66,y+15+i*19),fill='#13273b',outline=color,radius=3)
        for j in range(3): d.ellipse((x+8+j*9,y+6+i*19,x+11+j*9,y+9+i*19),fill=color)

def monitor(d,x,y,label,color=CYAN):
    rr(d,(x,y,x+84,y+57),fill=BG,outline=color,width=2,radius=5)
    d.line((x+42,y+57,x+42,y+68),fill=color,width=2)
    d.line((x+22,y+68,x+62,y+68),fill=color,width=2)
    for k in range(3): d.line((x+12,y+15+k*10,x+55-k*8,y+15+k*10),fill=color,width=2)
    text(d,(x+42,y+85),label,16,color,anchor='mm')

def plc(d,x,y,color=CYAN):
    rr(d,(x,y,x+88,y+70),fill='#15283b',outline=color,width=2,radius=7)
    text(d,(x+44,y+24),'PLC',22,color,True,anchor='mm')
    for i in range(6):
        xx=x+10+i*12
        d.rectangle((xx,y+51,xx+6,y+59),fill=color)
    for i in range(4): d.line((x+11+i*21,y-7,x+11+i*21,y),fill=color,width=2)

def tank(d,x,y,w,h,level,t,color=CYAN):
    rr(d,(x,y,x+w,y+h),fill=BG,outline=color,width=3,radius=12)
    level=max(.05,min(.95,level)); yy=y+h*(1-level)
    pts=[(x+4,y+h-5),(x+4,yy)]
    for i in range(w-7): pts.append((x+4+i,yy+2*math.sin(i*.07+t*1.8)))
    pts.append((x+w-4,y+h-5))
    d.polygon(pts,fill='#174f69' if color==CYAN else '#6b3442')
    d.line((x+1,yy,x+w-1,yy),fill=color,width=2)
    for p in [.25,.5,.75]: d.line((x+w-14,y+h*p,x+w-4,y+h*p),fill=MUTED,width=1)

def pump(d,x,y,t,color=CYAN):
    d.ellipse((x-22,y-22,x+22,y+22),fill=PANEL,outline=color,width=3)
    for i in range(3):
        a=t*2+i*2*math.pi/3
        d.line((x,y,x+16*math.cos(a),y+16*math.sin(a)),fill=color,width=4)
    d.ellipse((x-4,y-4,x+4,y+4),fill=color)

def network(d,t,focus='intro',scale=1):
    # Four fixed zones preserve orientation throughout the story.
    boxes=[(64,294,342,568),(360,294,522,568),(540,294,898,568),(916,294,1216,568)]
    labels=['ENTERPRISE IT','IT / OT BOUNDARY','OT CONTROL','PHYSICAL PROCESS']
    cols=[ORANGE if focus in ['it','boundary'] else CYAN,PURPLE,CYAN,GREEN]
    for box,label,col in zip(boxes,labels,cols):
        rr(d,box,fill=PANEL,outline=col if (focus=='intro' or label.startswith('OT') and focus=='recon') else LINE,width=1)
        text(d,(box[0]+16,box[1]+17),label,13,col,True,True)
    server(d,94,357,ORANGE if focus=='it' else CYAN)
    monitor(d,215,357,'Endpoint',ORANGE if focus=='it' else CYAN)
    d.line((158,390,214,390),fill=LINE,width=3)
    for yy in range(360,491,30):
        for xx in range(391,491,26):
            rr(d,(xx,yy,xx+22,yy+20),fill='#253041',outline=PURPLE,radius=2)
    monitor(d,573,357,'Engineering',CYAN)
    plc(d,754,358,CYAN)
    d.line((657,389,753,389),fill=LINE,width=3)
    d.line((298,389,382,389),fill=ORANGE if focus in ['it','boundary'] else LINE,width=3)
    d.line((499,389,572,389),fill=CYAN if focus in ['boundary','recon'] else LINE,width=3)
    d.line((842,389,953,389),fill=GREEN,width=3)
    d.line((954,389,954,459,1002,459),fill=GREEN,width=3)
    pump(d,989,459,t,GREEN)
    d.line((1011,459,1055,459),fill=GREEN,width=4)
    tank(d,1055,356,115,155,.58+.07*math.sin(t*.8),t,CYAN)
    text(d,(1069,532),'Flow + level',16,MUTED)
    if focus in ['intro','boundary','it']:
        for offset in [0,.34,.67]: packet(d,[(158,389),(382,389),(499,389),(614,389),(798,389),(954,389),(954,459),(1055,459)],t,ORANGE,.06,offset)
    else:
        for off in [0,.5]: packet(d,[(615,389),(798,389),(954,389),(954,459),(1055,459)],t,CYAN,.18,off)
    bottoms={'it':['Assumed starting condition','No intrusion footage'], 'boundary':['Defined research access','Identity + permissions'], 'recon':['Inspect','Read','Infer device roles'],'intro':['Digital access','Operational authority','Physical consequence']}
    vals=bottoms.get(focus,['Tool proposal','Validated execution','Observed consequence'])
    for i,val in enumerate(vals): pill(d,85+i*365,590,val,ORANGE if focus=='it' else CYAN)

def agent_loop(d,t):
    nodes=[(170,390,'OBSERVE','Process readings'),(464,390,'PLAN','Bounded context'),(756,390,'TOOL','Validated action'),(1050,390,'VERIFY','Runtime feedback')]
    active=int(t*1.5)%4
    for i,(x,y,name,sub) in enumerate(nodes):
        color=CYAN if i==active else MUTED
        rr(d,(x-107,y-67,x+107,y+86),fill=PANEL,outline=color,width=3 if i==active else 1)
        text(d,(x,y-30),f'0{i+1}',17,color,True,True,anchor='mm')
        text(d,(x,y+3),name,28,WHITE,True,anchor='mm')
        text(d,(x,y+48),sub,15,MUTED,anchor='mm')
        if i<3:
            d.line((x+107,y+3,x+187,y+3),fill=LINE,width=3)
            packet(d,[(x+107,y+3),(x+187,y+3)],t,CYAN,.8,i*.25)
    d.line((1050,477,1050,541,170,541,170,477),fill=CYAN,width=2)
    packet(d,[(1050,477),(1050,541),(170,541),(170,477)],t,CYAN,.2)
    pill(d,457,568,'EVIDENCE AT EVERY STEP',CYAN)

def process_scene(d,t,p,defense=False):
    rr(d,(64,290,647,602))
    rr(d,(674,290,1216,602))
    text(d,(91,310),'MODELED WATER PROCESS',16,CYAN,True,True)
    level=.43+.43*ease(p) if not defense else .86-.35*ease(p)
    color=RED if level>.73 else CYAN
    tank(d,370,358,158,191,level,t,color)
    pump(d,235,472,t,ORANGE if not defense else GREEN)
    d.line((94,472,213,472),fill=CYAN,width=5)
    d.line((257,472,370,472),fill=ORANGE if not defense else GREEN,width=5)
    for off in [0,.5]: packet(d,[(94,472),(370,472)],t,ORANGE if not defense else GREEN,.35,off)
    d.line((375,390,523,390),fill=RED,width=2)
    text(d,(378,371),'Safety threshold',13,RED)
    text(d,(158,544),'Pump / valve actions',18,MUTED)
    text(d,(702,311),'INDEPENDENT EVALUATION',16,CYAN,True,True)
    xx,yy,ww,hh=714,373,458,147
    for k in range(4): d.line((xx,yy+k*hh/3,xx+ww,yy+k*hh/3),fill=LINE,width=1)
    points1=[];points2=[]
    for i in range(140):
        q=i/139
        val=.43+(.43*ease(q*p) if not defense else .43-.35*ease(q*p))
        points1.append((xx+ww*q,yy+hh*(1-val)))
        points2.append((xx+ww*q,yy+hh*(1-(.47+.02*math.sin(q*9+t)))))
    d.line(points1,fill=RED if not defense else GREEN,width=4)
    d.line(points2,fill=CYAN,width=3)
    d.line((xx,yy+hh*.27,xx+ww,yy+hh*.27),fill=RED,width=1)
    pill(d,710,549,'PROCESS STATE',GREEN if defense else RED)
    pill(d,934,549,'DISPLAYED VALUE',CYAN)
    text(d,(86,623),'Illustrative curves and thresholds — not experimental results.',17,MUTED)

def historical(d,t,kind):
    col=ORANGE if kind=='stuxnet' else PURPLE if kind=='industroyer' else RED
    rr(d,(64,292,625,590),outline=col)
    rr(d,(652,292,1216,590))
    if kind=='stuxnet':
        text(d,(91,317),'TRUSTED ENGINEERING PATH',16,col,True,True)
        monitor(d,102,398,'Engineering',col)
        plc(d,317,399,col)
        d.line((185,429,317,429),fill=col,width=3)
        packet(d,[(185,429),(317,429)],t,col,.3)
        for j in range(3):
            x=493+j*33
            d.ellipse((x-11,401,x+11,474),outline=col,width=2)
            d.line((x,408,x+8*math.sin(t*4+j),466),fill=col,width=3)
        d.line((405,430,481,430),fill=col,width=3)
        wrapped(d,'A control system can execute the wrong physical intent.',683,331,487,31,WHITE,43,True)
        wrapped(d,'Research connection: preserve controller integrity and check process consequences independently.',683,456,480,23,MUTED,34)
    elif kind=='industroyer':
        text(d,(91,317),'INDUSTRIAL PROTOCOLS',16,col,True,True)
        plc(d,106,403,col)
        d.line((194,438,299,438),fill=col,width=3)
        for j in range(3):
            x=310+j*90
            y=385+j*18
            d.line((x,y,x,y+38),fill=col,width=3)
            angle=.55+.3*math.sin(t*1.2+j)
            d.line((x,y+38,x+30*math.sin(angle),y+38+30*math.cos(angle)),fill=col,width=4)
            d.line((x,y+75,x,y+120),fill=col,width=3)
        packet(d,[(194,438),(530,438)],t,col,.2)
        wrapped(d,'Protocol fluency can become operational authority.',683,331,487,31,WHITE,43,True)
        wrapped(d,'Research connection: distinguish a valid command from a safe, authorized process outcome.',683,456,480,23,MUTED,34)
    else:
        text(d,(91,317),'SAFETY INSTRUMENTED SYSTEM',16,col,True,True)
        cx,cy=345,441
        polygon=[(cx,cy-83),(cx+75,cy-50),(cx+62,cy+34),(cx,cy+86),(cx-62,cy+34),(cx-75,cy-50)]
        d.polygon(polygon,fill='#362332',outline=col,width=4)
        text(d,(cx,cy-4),'SIS',39,col,True,anchor='mm')
        for j in range(4):
            a=t*.4+j*math.pi/2
            x,y=cx+118*math.cos(a),cy+95*math.sin(a)
            d.ellipse((x-5,y-5,x+5,y+5),fill=col)
        wrapped(d,'Safety protection is an independent boundary.',683,331,487,31,WHITE,43,True)
        wrapped(d,'Research connection: assess safety state separately from confidence, intent, or tool success.',683,456,480,23,MUTED,34)
    text(d,(72,613),SOURCES[kind][0],16,col)
    text(d,(72,641),'Historical malware context; not an AI-agent incident reproduction.',16,MUTED)

def defense_scene(d,t,p):
    process_scene(d,t,p,True)
    for x,label in [(87,'OBSERVE'),(260,'CONTAIN'),(449,'RECOVER')]:
        pill(d,x,567,label,GREEN)

def evaluation(d,t):
    items=[('SAFETY','Constraint violations'),('RESPONSE','Detection + containment'),('AVAILABILITY','Downtime + recovery'),('RIGOR','Baselines + ablations')]
    for i,(title,desc) in enumerate(items):
        x=64+i*291
        rr(d,(x,310,x+267,568),outline=CYAN if i==int(t*.5)%4 else LINE)
        text(d,(x+21,333),f'0{i+1}',19,CYAN,True,True)
        text(d,(x+21,377),title,22,WHITE,True)
        wrapped(d,desc,x+21,420,223,20,MUTED,30)
        for j in range(9):
            bh=17+50*(.5+.5*math.sin(t*.7+j*.5+i))
            d.rounded_rectangle((x+22+j*25,545-bh,x+35+j*25,545),radius=3,fill=CYAN if i<2 else GREEN)
    text(d,(69,613),'Designs and instrumentation are implemented. Claims require matched run evidence.',18,MUTED)

def render(t):
    scene_index=next(i for i,s in enumerate(SCENES) if s[0]<=t<s[1]) if t<DURATION else len(SCENES)-1
    start,end,eyebrow,title,caption,kind=SCENES[scene_index]
    p=(t-start)/(end-start)
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    # Restrained moving grid gives the systems diagram depth without visual noise.
    for x in range(-40,W+40,64):
        xx=x+int(t*2)%64
        d.line((xx,0,xx,H),fill='#0e1826',width=1)
    for y in range(0,H,64): d.line((0,y,W,y),fill='#0e1826',width=1)
    text(d,(64,32),'ICS—AGENT',21,WHITE,True,True)
    text(d,(64,64),'CYBERSECURITY × AI × INDUSTRIAL CONTROL',12,MUTED,True,True)
    pill(d,981,36,'ONGOING RESEARCH',CYAN)
    d.line((64,96,1216,96),fill=LINE,width=1)
    text(d,(64,117),eyebrow,15,ORANGE if kind in SOURCES else CYAN,True,True)
    title_size=44 if len(title)<49 else 39
    text(d,(62,149),title,title_size,WHITE,True)
    wrapped(d,caption,65,211,1145,21,MUTED,30)
    if kind in ['intro','it','boundary','recon']: network(d,t,kind)
    elif kind=='loop': agent_loop(d,t)
    elif kind=='process': process_scene(d,t,p)
    elif kind in SOURCES: historical(d,t,kind)
    elif kind=='defense': defense_scene(d,t,p)
    elif kind=='evaluation': evaluation(d,t)
    elif kind=='outro':
        network(d,t,'intro')
        rr(d,(303,360,981,492),fill=BG,outline=CYAN,width=2,radius=20)
        text(d,(640,396),'Observe → Act → Evaluate',34,WHITE,True,anchor='mm')
        text(d,(640,453),'Controlled testbeds. Traceable decisions.',23,CYAN,anchor='mm')
    # Persistent context and playback progress.
    d.rectangle((0,677,W,H),fill=BG)
    text(d,(64,688),'CONCEPTUAL ANIMATION  /  AUTHORIZED RESEARCH TESTBEDS',12,MUTED,True,True)
    text(d,(1216,688),f'{int(t)//60:02d}:{int(t)%60:02d} / 02:30',12,MUTED,True,True,anchor='ra')
    for i,s in enumerate(SCENES):
        x1=64+s[0]/DURATION*1152; x2=64+s[1]/DURATION*1152-4
        d.rectangle((x1,670,x2,673),fill=CYAN if i<scene_index else LINE)
        if i==scene_index: d.rectangle((x1,670,mix(x1,x2,p),673),fill=CYAN)
    # A 0.35-second fade-through-dark between story beats.
    fade=min(1,p*(end-start)/.35,(1-p)*(end-start)/.35)
    if fade<1:
        base=Image.new('RGB',(W,H),BG)
        im=Image.blend(base,im,max(.12,fade))
    return im

def make_audio(path):
    sr=22050
    # Soft original ambient dyads: no voice, copyrighted music, or external samples.
    chord=[110.0,164.8138,220.0,261.6256]
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
        for second in range(DURATION):
            chunk=array('h')
            for n in range(sr):
                tt=second+n/sr
                env=min(1,tt/3,(DURATION-tt)/4)
                signal=sum(math.sin(2*math.pi*fr*tt)*(.28 if j==0 else .13) for j,fr in enumerate(chord))
                signal*=.018*env*(.85+.15*math.sin(tt*.4))
                chunk.append(int(signal*32767))
            f.writeframes(chunk.tobytes())

def write_support_files():
    captions=[]
    for i,(s,e,eye,title,caption,kind) in enumerate(SCENES,1):
        def ts(n): return f'{int(n)//3600:02d}:{int(n)%3600//60:02d}:{int(n)%60:02d},000'
        captions.append(f'{i}\n{ts(s)} --> {ts(e)}\n{title}\n{caption}\n')
    (ROOT/'overview.srt').write_text('\n'.join(captions))
    (ROOT/'storyboard.json').write_text(json.dumps([dict(start=s,end=e,chapter=eye,title=title,caption=cap,visual=kind) for s,e,eye,title,cap,kind in SCENES],indent=2)+'\n')
    (ROOT/'SOURCES.md').write_text('''# Research animation sources

The 2:30 film presents a conceptual IT → OT → physical-process storyline in authorized research testbeds. It is not captured exploitation footage. Its IT foothold and cross-boundary access are assumed narrative conditions, not a demonstrated capability claim for ICS-Agent. The historic incidents were malware campaigns, not AI-agent attacks, and the animation does not reproduce their malware.

- **Stuxnet (discovered 2010):** [ENISA, Stuxnet Analysis](https://www.enisa.europa.eu/news/enisa-news/stuxnet-analysis). Supports the description of Siemens industrial-control software and industrial-system targeting. [Ralph Langner, To Kill a Centrifuge (2013), primary technical analysis hosted by Yale](https://cs.yale.edu/homes/jf/Langner.pdf) supports the description of centrifuge-control manipulation; the film does not assert a damage count or reproduce the attack.
- **Industroyer (December 2016 incident; analysis published 2017):** [ESET Research, Industroyer retrospective](https://www.welivesecurity.com/2022/06/13/industroyer-cyber-weapon-brought-down-power-grid/). Supports the description of industrial-protocol use, substation equipment, and the Kyiv power outage.
- **TRITON (2017):** [UK NCSC, TRITON Malware Targeting Safety Controllers](https://www.ncsc.gov.uk/information/triton-malware-targeting-safety-controllers). Supports the description of Triconex safety-controller targeting. See also [CISA, HatMan safety-system malware analysis](https://www.cisa.gov/sites/default/files/documents/MAR-17-352-01%20HatMan%E2%80%94Safety%20System%20Targeted%20Malware_S508C.pdf).

The displayed process curves, thresholds, network topology, agent activity, and timing are illustrative animation data. They are not experimental results or details of an operational facility. The video uses original code-drawn visuals and an original synthesized ambient soundtrack, with no third-party image, video, or audio assets.

## Reproduce

Use Python with `Pillow` and `imageio-ffmpeg` installed, then run `python render_overview.py` in this directory. An existing `ffmpeg` on PATH can be used instead. `python render_overview.py --preview` produces just the poster/contact sheet. The renderer uses standard DejaVu fonts from the system, or a local `fonts/` folder, with a Pillow fallback.

Deliverables: `ics-agent-research-overview.mp4` (150 seconds), `overview-preview.gif`, `overview-poster.png`, `overview.srt`, and `storyboard.json`. Captions are burned into the film; the SRT is also provided for accessibility. The soundtrack contains no narration.
''')

def main():
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
    write_support_files()
    render(5).save(ROOT/'overview-poster.png')
    thumbs=[]
    for stamp in [5,17,30,44,59,77,90,103,115,128,140,148]:
        im=render(stamp).resize((480,270),Image.Resampling.LANCZOS)
        ImageDraw.Draw(im).text((12,249),f'{stamp//60:02d}:{stamp%60:02d}',font=font(14,True),fill=WHITE)
        thumbs.append(im)
    sheet=Image.new('RGB',(1440,1080),BG)
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*270))
    sheet.save(ROOT/'contact-sheet.jpg',quality=92)
    if args.preview:return
    ffmpeg=shutil.which('ffmpeg') or (imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg else None)
    if not ffmpeg: raise SystemExit('Install imageio-ffmpeg or provide ffmpeg on PATH.')
    output=ROOT/'ics-agent-research-overview.mp4'
    silent=ROOT/'_silent.mp4'
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(FPS*DURATION):
        process.stdin.write(render(i/FPS).tobytes())
        if i%(FPS*15)==0: print(f'Rendered {i/FPS:.0f}/{DURATION} seconds',flush=True)
    process.stdin.close()
    if process.wait()!=0:raise SystemExit('Video encoding failed')
    audio=ROOT/'_ambient.wav';make_audio(audio)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',str(audio),'-c:v','copy','-c:a','aac','-b:a','64k','-shortest','-movflags','+faststart',str(output)],check=True)
    # A compact 12-second original animation, linking from README to the full film.
    frames=[]
    for i in range(60):
        stamp=[3, 55, 73, 100][i//15]+(i%15)/5
        frames.append(render(stamp).resize((640,360),Image.Resampling.LANCZOS).convert('P',palette=Image.Palette.ADAPTIVE,colors=96))
    frames[0].save(ROOT/'overview-preview.gif',save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=False)
    silent.unlink();audio.unlink()
    print(json.dumps({'video':output.name,'duration':DURATION,'fps':FPS,'frames':FPS*DURATION,'width':W,'height':H,'files':{f.name:f.stat().st_size for f in ROOT.iterdir() if f.is_file()}},indent=2),flush=True)

if __name__=='__main__':main()
