# Instructor guide: all 8 topics

## One-sentence explanation for non-technical learners
1. Classical CV: apply hand-written image rules to find edges and shapes.
2. YOLO: find and label multiple objects in one pass.
3. Faster R-CNN: propose regions, then classify and refine each region.
4. Classification: choose a label for the whole image.
5. Segmentation: assign a label to each pixel.
6. Facial expression: classify visible facial-expression patterns; it cannot know a person's true feelings.
7. Siamese network: compare two inputs using learned feature representations.
8. GAN: generate new samples by training a generator against a discriminator.

## Recommended order
Start with classical OpenCV, then YOLO, Faster R-CNN, classification, segmentation, facial expression, Siamese network, and GAN.

## Questions to ask students
- What is the input and output of this model?
- Does it predict one label, boxes, pixel masks, similarity, or new images?
- What happens if lighting, background, scale, or image quality changes?
- What data would be needed to train for a new business use case?
- What mistakes could the model make, and what would the business impact be?
- How would you evaluate it beyond looking at one successful example?

## Demo safety and reliability
Use consented sample images. Do not infer identity, sensitive traits, or true internal feelings from a face. Avoid high-stakes decisions based on these classroom models. Check model/dataset licenses before redistribution or commercial deployment.
