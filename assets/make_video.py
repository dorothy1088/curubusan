"""14초짜리 업로드→예측 대시보드 스토리보드 MP4 생성 (예시만 사용)."""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math
from pathlib import Path
import numpy as np
import subprocess

W, H, FPS, SECONDS = 1280, 720, 15, 14
D = Path(__file__).resolve().parent
BOLD = '/usr/share/fonts/truetype/nanum/NanumSquareB.ttf'
REG = '/usr/share/fonts/truetype/nanum/NanumSquareR.ttf'
FALLBACK = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'

def font(sz, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else REG, sz)
    except OSError:
        return ImageFont.truetype(FALLBACK, sz)

F12 = font(13)
F15 = font(16)
F18 = font(19)
F20 = font(21, True)
F25 = font(27, True)
F36 = font(38, True)
F16B = font(16, True)

BG = (8, 24, 43)
WHITE = (238, 250, 255)
MUTED = (145, 177, 197)
TEAL = (65, 223, 207)
CYAN = (44, 173, 190)
GOLD = (243, 175, 101)
STROKE = (53, 91, 116)


def mix(a,b,k):
    return tuple(round((1-k)*a[i]+k*b[i]) for i in range(3))

def ease(x):
    x = max(0, min(1, x))
    return x*x*(3-2*x)

def vis(t,start,dur):
    return ease((t-start)/dur)

def overlay_alpha(im, rect, color, alpha):
    rgba = Image.new('RGBA', im.size, (0,0,0,0))
    rd = ImageDraw.Draw(rgba)
    rd.rounded_rectangle(rect, radius=15, fill=(*color, int(alpha*255)))
    im.paste(Image.alpha_composite(im.convert('RGBA'), rgba).convert('RGB'))

def polyline(draw, pts, fill, width=4):
    if len(pts)>=2:
        draw.line(pts, fill=fill, width=width, joint='curve')
    for px,py in pts:
        draw.ellipse((px-3,py-3,px+3,py+3), fill=fill)

# Background gradient drawn once.
base_arr=np.zeros((H,W,3),dtype=np.uint8)
for y in range(H):
    for x in range(W):
        d=math.hypot((x-W*.73)/W,(y-H*.25)/H)
        accent=max(0,1-d*1.8)
        a=(int(8+7*accent),int(24+19*accent),int(43+30*accent))
        base_arr[y,x]=a
BASE = Image.fromarray(base_arr,'RGB')


def draw_logo(draw,x,y):
    draw.ellipse((x-12,y-12,x+12,y+12),outline=TEAL,width=3)
    draw.arc((x-16,y-13,x+16,y+13),start=180,end=350,fill=TEAL,width=2)
    draw.line([(x,y-16),(x,y+17)],fill=TEAL,width=2)
    draw.line([(x-13,y+10),(x+13,y+10)],fill=TEAL,width=2)


def file_card(draw,cx,cy,label,sub,color,alpha=1,scale=1):
    x=int(cx-80*scale);y=int(cy-48*scale);ww=int(160*scale);hh=int(98*scale)
    fill=mix(BG,(21,62,82),.95)
    outline=mix(fill,color,min(.95,.72*alpha))
    draw.rounded_rectangle((x,y,x+ww,y+hh),radius=int(11*scale),fill=fill,outline=outline,width=2)
    draw.rounded_rectangle((x+int(10*scale),y+int(13*scale),x+int(40*scale),y+int(46*scale)),radius=6,fill=mix(fill,color,.25))
    draw.text((x+int(18*scale),y+int(22*scale)),"CSV",font=font(max(9,int(11*scale)), True), fill=color)
    draw.text((x+int(13*scale),y+int(53*scale)),label,font=font(max(12,int(15*scale)), True),fill=WHITE)
    draw.text((x+int(13*scale),y+int(76*scale)),sub,font=font(max(10,int(11*scale))),fill=MUTED)


def frame(t):
    im=BASE.copy(); d=ImageDraw.Draw(im)
    # header
    draw_logo(d,56,52)
    d.text((83,35),"BUSAN CRUISE  /  FORECAST FLOW",font=F16B,fill=TEAL)
    d.text((52,76),"자료를 넣으면, 20XX년 전망이 완성됩니다",font=F36,fill=WHITE)
    d.text((53,124),"원자료 + 입항 예정 스케줄 → 모델 학습 → 예측구간 → 월별 운영 검토",font=F18,fill=MUTED)

    # LEFT: person operator card
    d.rounded_rectangle((52,179,294,584),radius=21,fill=(13,39,60),outline=STROKE,width=2)
    d.text((75,202),"01  DATA INPUT",font=F16B,fill=TEAL)
    d.text((75,234),"담당자",font=F25,fill=WHITE)
    d.text((75,267),"CSV 자료 업로드",font=F15,fill=MUTED)
    # person torso chair
    d.ellipse((143,305,204,366),fill=(246,201,157))
    d.ellipse((152,332,202,355),fill=(24,46,67))
    d.rounded_rectangle((135,361,220,461),radius=23,fill=(32,148,157))
    d.polygon([(179,374),(192,393),(177,418),(163,393)],fill=(241,246,244))
    d.line([(201,388),(235,420),(254,402)],fill=(246,201,157),width=16)
    d.ellipse((247,393,265,411),fill=(246,201,157))
    d.rounded_rectangle((87,432,244,524),radius=10,fill=(19,55,76),outline=(73,127,147),width=3)
    d.rounded_rectangle((104,448,227,494),radius=5,fill=(10,31,49))
    d.text((118,459),"UPLOAD",font=F16B,fill=TEAL)
    d.rounded_rectangle((77,521,256,533),radius=6,fill=(78,127,145))
    d.text((89,552),"원본과 일정은 별개 입력",font=F12,fill=MUTED)

    # center upload tray
    d.rounded_rectangle((320,179,719,584),radius=20,fill=(12,36,55),outline=STROKE,width=2)
    d.text((343,202),"02  MODEL PIPELINE",font=F16B,fill=TEAL)
    d.text((343,231),"자동 처리 흐름",font=F25,fill=WHITE)
    d.rounded_rectangle((348,286,686,439),radius=18,fill=(14,46,65),outline=(66,123,142),width=2)
    for xx in range(362,686,20):
        d.line((xx,286,xx+9,286),fill=(62,124,142),width=2)
        d.line((xx,439,xx+9,439),fill=(62,124,142),width=2)
    d.text((390,303),"여기에 데이터를 올려주세요",font=F20,fill=WHITE)
    progress=vis(t,4.2,3.0)
    d.rounded_rectangle((382,396,655,408),radius=5,fill=(29,69,89))
    d.rounded_rectangle((382,396,382+int(273*progress),408),radius=5,fill=TEAL)
    if t<4.3:
        stxt="파일 입력을 확인하는 중"
    elif t<7.2:
        stxt="ETS 모델 학습 · 95% 구간 계산"
    else:
        stxt="입항수 · 승객수 예측 완료"
    d.text((383,458),stxt,font=F16B,fill=TEAL if t>=7.2 else MUTED)
    # Show both models explicitly so the 2023 common-start configuration is unambiguous.
    d.text((383,482),"입항수  ETS(A,Ad,A)",font=F16B,fill=WHITE)
    d.text((383,506),"승객수  ETS(A,Ad,A)",font=F16B,fill=WHITE)
    d.text((383,531),"공통 학습: 2023.01 ~ 최신 확정월",font=F12,fill=MUTED)
    d.text((383,549),"예측 결과 표시: 20XX년 1~12월",font=F12,fill=MUTED)

    # file drag motion (two files sequentially)
    q1=vis(t,1.0,2.7); q2=vis(t,2.2,2.7)
    x1=185+(320)*q1; y1=302-3*math.sin(math.pi*q1)+36*q1
    x2=185+(320)*q2; y2=304+35*q2
    if q1<.97:
        file_card(d,x1,y1,"원자료", "2023—최신 완료월",CYAN,1-q1*.15,.85)
    else:
        d.ellipse((465,354,483,372),fill=TEAL)
        d.line([(468,362),(473,368),(480,356)],fill=BG,width=2)
    if q2<.97:
        file_card(d,x2,y2+40,"20XX년 일정", "예정 입항·승객수",GOLD,1-q2*.15,.83)
    else:
        d.ellipse((605,354,623,372),fill=TEAL)
        d.line([(609,362),(614,368),(621,356)],fill=BG,width=2)
    if t>=4.6 and t<7.2:
        angle=t*5
        cx,cy=515,352
        d.ellipse((cx-31,cy-31,cx+31,cy+31),outline=(42,137,152),width=6)
        for k in range(7):
            xx=cx+27*math.cos(2*math.pi*k/7+angle)
            yy=cy+27*math.sin(2*math.pi*k/7+angle)
            d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=TEAL)
        d.ellipse((cx-7,cy-7,cx+7,cy+7),fill=TEAL)

    # right result dashboard
    d.rounded_rectangle((745,179,1228,584),radius=20,fill=(16,43,61),outline=STROKE,width=2)
    d.text((770,202),"03  20XX FORECAST",font=F16B,fill=TEAL)
    d.text((770,232),"월별 전망·비교 결과",font=F25,fill=WHITE)
    d.rounded_rectangle((771,286,1201,438),radius=13,fill=(10,34,51),outline=(36,73,98),width=1)
    for k in range(1,4):
        yy=294+33*k
        d.line((792,yy,1178,yy),fill=(30,66,85),width=1)
    d.text((792,300),"20XX  JAN — DEC",font=F12,fill=MUTED)
    points = []
    forecast_y=[392,377,372,350,345,349,332,327,341,310,319,295]
    n = max(0, min(12,int(12*vis(t,7.2,2.6))))
    xs = np.linspace(798,1169,12)
    for i in range(n):
        yy=forecast_y[i]
        points.append((int(xs[i]),yy))
    if n>1:
        upper=[(x,y-19) for x,y in points]
        lower=[(x,y+19) for x,y in reversed(points)]
        layer=Image.new('RGBA',(W,H),(0,0,0,0)); ld=ImageDraw.Draw(layer)
        ld.polygon(upper+lower,fill=(*TEAL,38))
        im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB');d=ImageDraw.Draw(im)
    polyline(d,points,TEAL,width=4)
    # results table cascade
    reveal=vis(t,9.5,2.1)
    d.text((771,451),"월별 비교표",font=F16B,fill=WHITE if reveal>0 else (20,49,63))
    x_starts=[780,880,1024]
    heads=["연월", "예측 입항", "예측 승객"]
    if reveal>0:
        d.rounded_rectangle((771,481,1202,514),radius=6,fill=(32,76,92))
        for x,h in zip(x_starts,heads):d.text((x,490),h,font=F12,fill=TEAL)
        rows=min(3,int(4*reveal))
        for j in range(rows):
            y=516+j*20
            d.text((783,y),f"20XX-{j+1:02d}",font=F12,fill=WHITE)
            d.rounded_rectangle((890,y+5,968-j*3,y+11),radius=3,fill=(43,150,155))
            d.rounded_rectangle((1030,y+5,1154-j*4,y+11),radius=3,fill=(69,140,165))

    # connector arrow between pipeline and dashboard, pulse
    if t>6.7:
        k=vis(t,6.7,.9)
        d.line((711,364,743,364),fill=mix((60,100,117),TEAL,k),width=4)
        d.polygon([(738,357),(750,364),(738,371)],fill=TEAL)

    # bottom pipeline indicators
    labels=["원자료 업로드", "결측·기간 검증", "ETS 시뮬레이션", "20XX 결과 생성"]
    done=[1.0,4.1,7.1,10.5]
    for k,lab in enumerate(labels):
        x=52+k*300
        active=t>=done[k]
        d.rounded_rectangle((x,615,x+280,674),radius=14,fill=(19,58,76) if active else (15,40,60),outline=TEAL if active else (41,82,108),width=2)
        d.ellipse((x+13,631,x+38,657),fill=TEAL if active else (34,70,90))
        d.text((x+21,636),str(k+1),font=F12,fill=BG if active else MUTED)
        d.text((x+50,633),lab,font=F15,fill=WHITE if active else MUTED)
    if t>=11.3:
        k=vis(t,11.3,1.1)
        d.rounded_rectangle((756,98,1228,151),radius=15,fill=mix(BG,(15,80,80),k),outline=TEAL,width=2)
        d.text((775,112),"완료  |  20XX.01—12 전망 준비",font=F20,fill=TEAL)
    d.text((52,692),"* 시뮬레이션 과정 안내용 애니메이션 · 그래프는 개념도이며 실제 예측 수치가 아닙니다",font=F12,fill=(112,147,166))
    return im


def main():
    video = D/'workflow_demo.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
         '-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','-',
         '-an','-c:v','libx264','-preset','ultrafast','-crf','24',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(FPS*SECONDS):
        im=frame(i/FPS)
        if i==FPS*11:
            im.save(D/'workflow_preview.png')
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    err=proc.stderr.read().decode('utf8',errors='replace')
    rc=proc.wait()
    if rc:
        raise RuntimeError('Video encoding failed: '+err)
    print('Wrote',video,video.stat().st_size,'bytes')

if __name__=='__main__':main()
