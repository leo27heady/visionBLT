import random

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from shapekit import Scene, SceneType, Random2DShapeCreator

from bytelatent.args import DataConfig, TemporalPatterns, IntervalModel
from bytelatent.data.data_types import VisionBatch


class ShapeDataset:
    def __init__(self, device, config: DataConfig, start_step: int, steps: int):
        self.scale_factor = 1000
        self.device = device
        self.config = config
        self.current_steps = start_step
        self.total_steps = steps
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.int64),
            # transforms.Normalize(mean=[0.5], std=[0.5])  # normalize later
        ])
        self.time_to_pred_min = self.config.time_to_pred.min

        assert 0 <= self.config.min_patterns <= len(self.config.temporal_patterns)
        
        if self.config.gradual_complexity and self.config.time_to_pred.max != 0:
            assert sum(self.config.gradual_complexity) == 1.0
            assert len(self.config.gradual_complexity) == self.config.time_to_pred.max - self.time_to_pred_min + 1
            
            self.increase_complexity_step_iter = iter(self.config.gradual_complexity)
            self.increase_complexity_steps = 0
            self.start_max_batch_time = self.time_to_pred_min - 1
            
            while self.increase_complexity_steps <= self.current_steps:
                self.increase_complexity_steps += next(self.increase_complexity_step_iter) * self.total_steps
                self.start_max_batch_time += 1
        else:
            self.increase_complexity_steps = None
            self.start_max_batch_time = self.config.time_to_pred.max
    
    def scale(self, var: int | float) -> int:
        return int(var * self.scale_factor)
    
    def rgb2gray(self, rgb_img: np.ndarray) -> np.ndarray:
        gray_img = np.dot(rgb_img, [0.2989, 0.5870, 0.1140])
        gray_img = np.expand_dims(gray_img, axis=-1)
        return np.round(gray_img).astype(int)

    def create_iter(self):
        triangle = Random2DShapeCreator().create_equilateral_triangle()
        
        scene = Scene(
            triangle, SceneType.DIM_2, self.config.render_window_size,
            bg_color="white", 
            mesh_color="black",
            show_edges=False,
            lighting=False,
            line_width=4.0,
            fixed_camera_distance=2.7, 
            axis="z"
        )
        
        while True:

            if self.increase_complexity_steps and self.current_steps >= self.increase_complexity_steps:
                if self.start_max_batch_time == 0:
                    self.time_to_pred_min = 1  # if start from 0 then increase min to 1
                
                if self.config.time_to_pred.max == self.start_max_batch_time:
                    self.increase_complexity_steps = 0
                else:
                    self.start_max_batch_time += 1
                    self.increase_complexity_steps = self.current_steps + next(self.increase_complexity_step_iter, 0) * self.total_steps

            current_batch_time = random.randint(self.time_to_pred_min, self.start_max_batch_time)
            
            batch_images_x, batch_images_y, batch_angles, batch_temp_patterns = [], [], [], []
            for batch in range(self.config.batch_size):
                
                # Selected imperially to fit the triangle to the render window, should be adjusted carefully
                base = random.randint(self.scale(0.5), self.scale(1)) / self.scale_factor
                shift = random.randint(self.scale(-0.2), self.scale(base + 0.2)) / self.scale_factor
                height = random.randint(self.scale(0.5), self.scale(1)) / self.scale_factor
                figure = Random2DShapeCreator().create_triangle(base, shift, height)
                
                # Prepare scene with the new figure, memory efficient - the scene remains the same
                scene.prepare_scene(
                    figure,
                    bg_color="white", 
                    mesh_color="black",
                    show_edges=False,
                    lighting=False,
                    line_width=4.0,
                    distance_factor=1.0,
                    fixed_camera_distance=2.7, 
                    axis="z"
                )
                figure.rotate_z(random.randint(0, self.scale(360)) / self.scale_factor, point=scene.center_of_mass, inplace=True)
                scene.plotter.render()

                angle = random.randint(self.scale(self.config.angle.min), self.scale(self.config.angle.max)) / self.scale_factor
                step = random.choice((-angle, angle))

                selected_temporal_patterns = self.config.temporal_patterns.copy()
                if TemporalPatterns.ACCELERATION in self.config.temporal_patterns and TemporalPatterns.DECELERATION in self.config.temporal_patterns:
                    if random.choice((True, False)):
                        selected_temporal_patterns.remove(TemporalPatterns.ACCELERATION)
                    else:
                        selected_temporal_patterns.remove(TemporalPatterns.DECELERATION)
                
                if TemporalPatterns.OSCILLATION in self.config.temporal_patterns and TemporalPatterns.INTERRUPTION in self.config.temporal_patterns:
                    if random.choice((True, False)):
                        selected_temporal_patterns.remove(TemporalPatterns.OSCILLATION)
                    else:
                        selected_temporal_patterns.remove(TemporalPatterns.INTERRUPTION)

                # Can be empty
                if selected_temporal_patterns:
                    selected_temporal_patterns = np.random.choice(
                        selected_temporal_patterns, 
                        size=random.randint(self.config.min_patterns, len(selected_temporal_patterns) if self.config.pattern_combining else 1),
                        replace=False
                    )
                
                acceleration, deceleration, oscillation_period, interruption_period = 0.0, 0.0, 0.0, 0.0
                step_swap = 0.0
                
                if TemporalPatterns.OSCILLATION in selected_temporal_patterns:
                    oscillation_period = random.choice(list(range(self.config.oscillation_period.min, self.config.oscillation_period.max + 1)))
                elif TemporalPatterns.INTERRUPTION in selected_temporal_patterns:
                    interruption_period = random.choice(list(range(self.config.interruption_period.min, self.config.interruption_period.max + 1)))
                
                if TemporalPatterns.ACCELERATION in selected_temporal_patterns:
                    step /= 1.5
                    acceleration = 1.0 + random.randint(self.scale(self.config.acceleration_hundredth.min / 2), self.scale(self.config.acceleration_hundredth.max / 2)) / (self.scale_factor * 100)
                elif TemporalPatterns.DECELERATION in selected_temporal_patterns:
                    step *= 1.5
                    deceleration = 1.0 + random.randint(self.scale(self.config.acceleration_hundredth.min * 2), self.scale(self.config.acceleration_hundredth.max * 2)) / (self.scale_factor * 100)
                    deceleration = 1 / deceleration

                images, angles = [], []
                for i in range(self.config.context_size + current_batch_time):
                    if step != 0.0:
                        figure.rotate_z(step, point=scene.center_of_mass, inplace=True)
                        scene.plotter.render()

                    image = np.array(scene.plotter.screenshot())
                    image = self.rgb2gray(image)
                    images.append(torch.Tensor(self.transform(image)).to(device=self.device))
                    angles.append(step)

                    # Multiple patterns at a time
                    if TemporalPatterns.OSCILLATION in selected_temporal_patterns:
                        if (i + 1) % oscillation_period == 0:
                            step *= -1.0
                    elif TemporalPatterns.INTERRUPTION in selected_temporal_patterns:
                        if (i + 1) % interruption_period == 0:
                            step, step_swap = step_swap, step

                    if TemporalPatterns.ACCELERATION in selected_temporal_patterns:
                        step *= acceleration
                    elif TemporalPatterns.DECELERATION in selected_temporal_patterns:
                        step *= deceleration

                batch_images = torch.stack(images)
                batch_images_x.append(batch_images[:-current_batch_time])
                batch_images_y.append(batch_images[current_batch_time:])

                batch_angles.append(torch.Tensor(angles))
                batch_temp_patterns.append(torch.Tensor([acceleration, deceleration, oscillation_period, interruption_period]))
            
            self.current_steps += 1

            yield VisionBatch(
                x=torch.stack(batch_images_x),
                y=torch.stack(batch_images_y),
                batch_t=torch.arange(start=1, end=current_batch_time + 1).to(device=self.device),
                angles=torch.stack(batch_angles).to(device=self.device),
                temp_patterns=torch.stack(batch_temp_patterns).to(device=self.device)
            )
