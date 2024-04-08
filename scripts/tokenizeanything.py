import os
import torch
import numpy as np
import gradio as gr
import gradio_image_prompter as gr_ext
from modules import script_callbacks
from tokenize_anything.engine.infer import Inference

example_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
weights = os.path.join(example_path, "models")
concept_weights = os.path.join(example_path, "concepts")

infer = Inference(model_type="tap_vit_l", weights=weights, concept_weights=concept_weights)


def tokenize_anything(click_img, mask_img, prompt, multipoint):
    img, point = None, np.array([[[0, 0, 4]]])
    if prompt == 0 and click_img is not None:
        img, points = click_img["image"], click_img["points"]
        points = np.array(points).reshape((-1, 2, 3))
        if multipoint == 1:
            points = points.reshape((-1, 3))
            lt = points[np.where(points[:, 2] == 2)[0]][None, :, :]
            rb = points[np.where(points[:, 2] == 3)[0]][None, :, :]
            poly = points[np.where(points[:, 2] <= 1)[0]][None, :, :]
            points = [lt, rb, poly] if len(lt) > 0 else [poly, np.array([[[0, 0, 4]]])]
            points = np.concatenate(points, axis=1)

    elif prompt == 1 and mask_img is not None:
        img, points = mask_img["background"], []
        for layer in mask_img["layers"]:
            ys, xs = np.nonzero(layer[:, :, 0])
            if len(ys) > 0:
                keep = np.linspace(0, ys.shape[0], 11, dtype="int64")[1:-1]
                points.append(np.stack([xs[keep][None, :], ys[keep][None, :]], 2))
        if len(points) > 0:
            points = np.concatenate(points).astype("float32")
            points = np.pad(points, [(0, 0), (0, 0), (0, 1)], constant_values=1)
            pad_points = np.array([[[0, 0, 4]]], "float32").repeat(points.shape[0], 0)
            points = np.concatenate([points, pad_points], axis=1)

    img = img[:, :, (2, 1, 0)] if img is not None else img
    img = np.zeros((480, 640, 3), dtype="uint8") if img is None else img
    points = np.array([[[0, 0, 4]]]) if (len(points) == 0 or points.size == 0) else points
    inputs = {"img": img, "points": points.astype("float32")}

    outputs = infer.run(inputs)

    scores, masks = outputs["scores"], outputs["masks"]
    concepts, captions = outputs["concepts"], outputs["captions"]
    text_template = "{} ({:.2f}, {:.2f}): {}"
    text_contents = concepts, scores[:, 0], scores[:, 1], captions
    texts = np.array([text_template.format(*vals) for vals in zip(*text_contents)])
    annotations = [(x, y) for x, y in zip(masks, texts)]
    return inputs["img"][:, :, ::-1], annotations


def on_ui_tabs():
    def on_reset_btn():
        click_img, draw_img = gr.Image(None), gr.ImageEditor(None)
        anno_img = gr.AnnotatedImage(None)
        return click_img, draw_img, anno_img

    with gr.Blocks(analytics_enabled=False) as tokenize_anything_interface:
        title = "Tokenize Anything"
        header = (
            "<div align='center'>"
            "<h1>Tokenize Anything via Prompting</h1>"
            "</div>"
        )
        theme = "soft"
        css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
                 #anno-img .mask.active {opacity: 0.7}"""

        app, _ = gr.Blocks(title=title, theme=theme, css=css).__enter__(), gr.Markdown(header)
        container, column = gr.Row().__enter__(), gr.Column().__enter__()
        click_tab, click_img = gr.Tab("Point+Box").__enter__(), gr_ext.ImagePrompter(show_label=False)
        interactions = "LeftClick (FG) | MiddleClick (BG) | PressMove (Box)"
        gr.Markdown("<h3 style='text-align: center'>[🖱️ | 🖐️]: 🌟🌟 {} 🌟🌟 </h3>".format(interactions))
        point_opt = gr.Radio(["Batch", "Ensemble"], label="Multipoint", type="index", value="Batch")

        _, draw_tab = click_tab.__exit__(), gr.Tab("Sketch").__enter__()
        draw_img, _ = gr.ImageEditor(show_label=False), draw_tab.__exit__()
        prompt_opt = gr.Radio(["Click", "Draw"], type="index", visible=False, value="Click")
        row, reset_btn, submit_btn = gr.Row().__enter__(), gr.Button("Reset"), gr.Button("Execute")
        _, _, column = row.__exit__(), column.__exit__(), gr.Column().__enter__()
        anno_img = gr.AnnotatedImage(elem_id="anno-img", show_label=False)

        reset_btn.click(on_reset_btn, [], [click_img, draw_img, anno_img])
        submit_btn.click(tokenize_anything, [click_img, draw_img, prompt_opt, point_opt], [anno_img])
        click_tab.select(lambda: "Click", [], [prompt_opt])
        draw_tab.select(lambda: "Draw", [], [prompt_opt])
        column.__exit__(), container.__exit__(), app.__exit__()

    return [(tokenize_anything_interface, "TokanizeAnything", "TokenizeAnything")]


script_callbacks.on_ui_tabs(on_ui_tabs)
