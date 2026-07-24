import { useEffect } from 'react'
import { LandingNav } from '../components/landing/LandingNav'
import { Hero } from '../components/landing/Hero'
import { Stats, Features, HowItWorks, Impact, CTA, Footer } from '../components/landing/Sections'

export default function Landing() {
  useEffect(() => {
    document.documentElement.style.scrollBehavior = 'smooth'
    return () => {
      document.documentElement.style.scrollBehavior = ''
    }
  }, [])

  return (
    <div className="min-h-screen bg-cv-bg">
      <LandingNav />
      <main>
        <Hero />
        <Stats />
        <Features />
        <HowItWorks />
        <Impact />
        <CTA />
      </main>
      <Footer />
    </div>
  )
}
