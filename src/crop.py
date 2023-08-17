# %% [markdown]
# Primeiro fazemos os imports

# %%
import cv2
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# O algoritmo assume que todas as imagens foram coletadas utilizando o mesmo aparelho e a mesma configuração, isso garante que a região de interesse tenha o mesmo tamanha e posição em todas as imagens.
# 
# Dito isso a primeira etapa do algoritimo é carregar todas as imagens que se deseja recortar a area de interesse respeitando as retrições descritas acima.

# %%
images = [np.asarray(Image.open(img)) for img in glob.glob("./imagens_para_recorte/*")]

# %% [markdown]
# Uma vez que todas as imagens foram carregadas fazemos um teste e marcamos de Branco todos os pixels que tem o valor diferente de 0.
# 
# Lembrando que em uma imagem os tons mais escuros tem valor que tende a 0 sendo 0 o tom mais escuro.

# %%
masks = [(img > 0).astype('f4') for img in images]

# %% [markdown]
# No próximo passo todas as mascaras são somadas em uma unica imagem e por fim os valores dos pixels são limitados entre 0 e 1.
# 
# $$
#     Se\ p\ >\ 1\ então\ p\ =\ 1;
# $$
# $$
#     Se\ p\ <\ 0\ então\ p\ =\ 0;  
# $$

# %%
final_mask = np.clip(np.sum(np.array(masks, dtype='f4'),axis=0), 0, 1)

plt.imshow(final_mask)

# %% [markdown]
# Continuando, é utilizado uma função que procura os contornos de todos objetos na mascara final.
# 
#     Obs: Os contornos são pintados em verde apenas para observação.

# %%
cnts, _ = cv2.findContours(
    cv2.cvtColor((final_mask * 255).astype("u1"), cv2.COLOR_BGR2GRAY),
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_SIMPLE,
)[-2:]



draw = (final_mask * 255).astype("u1")
cv2.drawContours(draw, cnts, -1, (0,255,0), 3)
plt.imshow(draw)

# %% [markdown]
# Dentre todos os contornos encontrados o contorno com maior area é selecionado.

# %%
c = max(cnts, key=cv2.contourArea)


draw = (final_mask * 255).astype("u1")
cv2.drawContours(draw, [c], -1, (0,255,0), 5)
plt.imshow(draw)

# %% [markdown]
# Por fim, as coordenadas do retangulo que melhor se encaixa no contorno encontrado são as coordenadas que devemos recortar todas as imagens.

# %%
x, y, w, h = cv2.boundingRect(c)
x, y, w, h

# %% [markdown]
# E finalmente todas as imagens são recortadas com as mesmas coordenadas.

# %%
croped = [img[y : y + h, x : x + w] for img in images]

# %%
for i, img in enumerate(croped):
    cv2.imwrite(f"MONTAGEM/02/croped_{i}.png", img)


