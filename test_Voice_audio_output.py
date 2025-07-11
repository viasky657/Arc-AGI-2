from contineous_thought_machines.models.ctm_Diffusion_NEWNEW import EnhancedCTMConfig, EnhancedCTMDiffusion

config = EnhancedCTMConfig()
model = EnhancedCTMDiffusion(config)

text, audio = model.generate_text_and_audio_simultaneously("Hello world")

print("Generated Text:", text)
print("Audio Shape:", audio.shape)